#!/usr/bin/env python3
"""
Optimized hand-sign feature extractor + SVM classifier
- incremental per-image caching (md5 hash)
- pathlib usage
- ThreadPoolExecutor for concurrency
- reuse grayscale image for feature extractors
- sklearn Pipeline + cross-val + proper misclassification mapping
- logging + timing + optional visualization
"""

import os
import cv2
import numpy as np
import pandas as pd
import hashlib
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Config / Global Settings -----------------
KERNEL_5 = np.ones((5, 5), np.uint8)
CACHE_FILE = "features_cache_svm_solidity_v3.csv"
MODEL_FILE = "svm_model_joblib.pkl"

THRESHOLDS = {
    "finger_area_min": 300,
    "finger_area_max": 5000,
    "defect_depth_min": 1000,
    "enclosed_blob_area_min": 50000,
    "solidity_thresh": 0.0  # not used as binary but kept for future
}

# set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ----------------- Utilities -----------------
def md5_of_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


# ----------------- Image helpers (single grayscale reuse) -----------------
def load_image_cv(image_path: Path):
    img = cv2.imread(str(image_path))
    return img


def morph_cleanup(mask):
    # common close then open
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_5), cv2.MORPH_OPEN, KERNEL_5)


def isolate_hand_sign(image):
    """Isolate non-green parts (assumes green background). Returns color image with background blacked out."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = morph_cleanup(green_mask)
    mask_inverted = cv2.bitwise_not(green_mask)
    isolated = cv2.bitwise_and(image, image, mask=mask_inverted)
    return isolated


# ----------------- Feature extraction functions (operate on grayscale or color when needed) -----------------
def extract_finger_features_from_gray(gray):
    """Compute average finger spacing based on convex hull bounding rect dims for candidate contours."""
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dims = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        aspect_ratio = float(area) / (perimeter + 1e-8)
        if (aspect_ratio < 0.8) and (THRESHOLDS["finger_area_min"] < area < THRESHOLDS["finger_area_max"]):
            try:
                hull = cv2.convexHull(c)
                x, y, w, h = cv2.boundingRect(hull)
                dims.append((w, h))
            except Exception:
                continue

    if not dims:
        return 0.0
    dims_np = np.array(dims, dtype=np.float32)
    diag = np.sqrt(np.sum(dims_np ** 2, axis=1))
    return float(np.mean(diag))


def has_enclosed_blob_from_gray(gray):
    """Detect whether there is a large enclosed child contour (using RETR_CCOMP).
    Returns 1 if largest enclosed child contour area > threshold, else 0."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure object is white on black for consistent findContours use
    if np.mean(binary > 127) > 0.5:
        binary = cv2.bitwise_not(binary)

    binary = morph_cleanup(binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    hierarchy_arr = hierarchy[0]
    enclosed_areas = []
    for i, h in enumerate(hierarchy_arr):
        # if this contour has a parent (h[3] != -1) it's enclosed
        if h[3] != -1:
            try:
                area = cv2.contourArea(contours[i])
                enclosed_areas.append(area)
            except Exception:
                continue
    if not enclosed_areas:
        return 0
    largest = max(enclosed_areas)
    return 1 if largest > THRESHOLDS["enclosed_blob_area_min"] else 0


def count_convexity_defects_from_gray(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 4:
        return 0
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        return 0
    if defects is None:
        return 0
    # defects array: (start_idx, end_idx, far_idx, fixpt_depth)
    # depth scaled by 256 in some OpenCV builds; we compare to threshold
    valid_defects = [d for d in defects if d[0][3] > THRESHOLDS["defect_depth_min"]]
    return len(valid_defects)


def calculate_solidity_from_gray(gray):
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area) / float(hull_area)


# ----------------- Per-image processing & caching -----------------
def process_single_image_tuple(args):
    """Wrapper for ThreadPoolExecutor: args is (class_name, image_path)"""
    class_name, image_path = args
    return process_single_image(class_name, Path(image_path))


def process_single_image(class_name: str, image_path: Path):
    """Load image, isolate hand, compute features. Returns dict including ImagePath and Hash."""
    try:
        img = load_image_cv(image_path)
        if img is None:
            logging.warning(f"Unable to read image: {image_path}")
            return None

        isolated = isolate_hand_sign(img)

        # reuse gray across feature functions
        gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)

        features = {}
        features["Class"] = class_name
        features["ImagePath"] = str(image_path)
        features["Hash"] = md5_of_file(image_path)

        features["FingerSpacing"] = extract_finger_features_from_gray(gray)
        features["HasEnclosedBlob"] = has_enclosed_blob_from_gray(gray)
        features["DefectCount"] = count_convexity_defects_from_gray(gray)
        features["Solidity"] = calculate_solidity_from_gray(gray)

        return features
    except Exception as e:
        logging.exception(f"Error processing {image_path}: {e}")
        return None


def analyze_features(data_folder: Path, visualize: bool = True, force_recompute: bool = False):
    """
    Walk dataset directory structure:
    data_folder/
       class_a/
         img1.jpg
         ...
       class_b/
         ...
    Returns DataFrame of features; caches results to CSV and uses per-image hash to avoid reprocessing.
    """
    start = time.perf_counter()
    data_folder = Path(data_folder)
    if not data_folder.exists() or not data_folder.is_dir():
        raise ValueError(f"data_folder does not exist or not a directory: {data_folder}")

    # Load existing cache (if present)
    cache_df = None
    if Path(CACHE_FILE).exists() and not force_recompute:
        try:
            cache_df = pd.read_csv(CACHE_FILE)
            logging.info(f"Loaded existing cache with {len(cache_df)} entries from '{CACHE_FILE}'.")
        except Exception:
            logging.warning("Failed to read existing cache, will recompute.")
            cache_df = None

    # collect all image paths
    all_images = []
    class_dirs = [p for p in data_folder.iterdir() if p.is_dir()]
    for class_dir in class_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in class_dir.glob(ext):
                all_images.append((class_dir.name, str(img_path)))

    total_images = len(all_images)
    logging.info(f"Found {total_images} images across {len(class_dirs)} classes.")

    # If cache exists, determine which images to process (new/changed)
    to_process = []
    existing_lookup = {}
    if cache_df is not None:
        # create lookup by ImagePath -> Hash
        existing_lookup = dict(zip(cache_df["ImagePath"].astype(str), cache_df["Hash"].astype(str)))
    for class_name, img_path in all_images:
        p = str(img_path)
        if force_recompute or cache_df is None:
            to_process.append((class_name, img_path))
        else:
            try:
                current_hash = md5_of_file(Path(img_path))
            except Exception:
                to_process.append((class_name, img_path))
                continue
            cached_hash = existing_lookup.get(p)
            if cached_hash != current_hash:
                to_process.append((class_name, img_path))
            else:
                # no change, skip
                pass

    logging.info(f"{len(to_process)} images to (re)process; using threads with os.cpu_count() workers.")

    records = []
    # Start with records from cache (if any and not force recompute)
    if cache_df is not None and not force_recompute:
        records = cache_df.to_dict("records")

    # process images concurrently
    if to_process:
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as exe:
            futures = {exe.submit(process_single_image_tuple, args): args for args in to_process}
            i = 0
            for fut in as_completed(futures):
                i += 1
                args = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logging.exception(f"Error in future for {args}: {e}")
                    res = None
                if res:
                    # replace any existing cached entry for same ImagePath
                    records = [r for r in records if r.get("ImagePath") != res["ImagePath"]]
                    records.append(res)
                if i % 10 == 0 or i == len(futures):
                    logging.info(f"  Processed {i}/{len(futures)} (in current run)...")

    df = pd.DataFrame(records)
    # ensure columns and types
    if df.empty:
        raise RuntimeError("No valid images processed; resulting dataframe is empty.")
    # convert numeric columns
    for col in ("FingerSpacing", "HasEnclosedBlob", "DefectCount", "Solidity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Save cache
    df.to_csv(CACHE_FILE, index=False)
    elapsed = time.perf_counter() - start
    logging.info(f"Finished processing features ({len(df)} items). Time: {elapsed:.1f}s. Cache saved to '{CACHE_FILE}'.")
    if visualize:
        visualize_features(df)
    return df


# ----------------- Visualization -----------------
def visualize_features(df):
    try:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x='Class', y='FingerSpacing', palette='Set2')
        plt.title("Finger Spacing by Class")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='Class', hue='HasEnclosedBlob', palette='Set1')
        plt.title("Enclosed Blob Presence by Class")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Class', y='DefectCount', palette='pastel')
        plt.title("Convexity Defects Count by Class")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Class', y='Solidity', palette='coolwarm')
        plt.title("Solidity by Class")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        for c in df['Class'].unique():
            subset = df[df['Class'] == c]
            plt.scatter(subset['FingerSpacing'], subset['DefectCount'], label=c, s=50, alpha=0.7)
        plt.xlabel("Average Finger Spacing")
        plt.ylabel("Convexity Defects Count")
        plt.title("2D Scatter Plot: Finger Spacing vs Convexity Defects")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        logging.exception("Visualization failed (likely in headless environment). Skipping plots.")


# ----------------- SVM training & evaluation -----------------
def evaluate_svm(df, save_model: bool = True, do_cross_val: bool = True):
    """
    df must contain columns: Class, FingerSpacing, HasEnclosedBlob, DefectCount, Solidity, ImagePath
    """
    # Prepare X, y
    features = ['FingerSpacing', 'HasEnclosedBlob', 'DefectCount', 'Solidity']
    X_df = df[features].copy()
    y = df['Class'].values

    # Label encode for class names
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Use pipeline (scaler -> SVC)
    pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=5, gamma='scale', decision_function_shape='ovr', probability=False))

    # Cross-validation
    if do_cross_val:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(pipe, X_df.values, y_encoded, cv=cv, n_jobs=-1)
            logging.info(f"Cross-val accuracy (5-fold): mean={scores.mean():.4f}, std={scores.std():.4f}")
        except Exception:
            logging.exception("Cross-validation failed. Continuing with single train/test split.")

    # Train/test split but keep dataframe mapping for misclassification analysis
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Class'])
    X_train = df_train[features].values
    X_test = df_test[features].values
    y_train = le.transform(df_train['Class'].values)
    y_test = le.transform(df_test['Class'].values)

    # Fit
    pipe.fit(X_train, y_train)

    # Optionally save model + label encoder and feature names
    if save_model:
        try:
            joblib.dump({"pipeline": pipe, "label_encoder": le, "features": features}, MODEL_FILE)
            logging.info(f"Saved trained model to '{MODEL_FILE}'.")
        except Exception:
            logging.exception("Failed to save model to disk.")

    # Predict and evaluate
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_, square=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SVM Confusion Matrix (with Solidity Feature)")
    plt.tight_layout()
    plt.show()

    logging.info("\nClassification Report:\n")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # Misclassification analysis: map back properly to df_test rows
    mis_mask = (y_pred != y_test)
    if mis_mask.any():
        mis_df = df_test.reset_index(drop=True).loc[mis_mask]
        mis_pred = le.inverse_transform(y_pred[mis_mask])
        mis_actual = le.inverse_transform(y_test[mis_mask])
        logging.info(f"Found {len(mis_df)} misclassified samples:")
        for idx, row in mis_df.iterrows():
            # idx corresponds to position in mis_df; find corresponding predicted/actual
            i = idx  # since we reset_index, idx corresponds
            # But safer: iterate zipped
            pass
        # iterate properly:
        for (r, pred, actual) in zip(mis_df.itertuples(index=False), mis_pred, mis_actual):
            print(f"Predicted: {pred:15s} | Actual: {actual:15s} | Image: {r.ImagePath} | FingerSpacing: {r.FingerSpacing:.2f} | EnclosedBlob: {int(r.HasEnclosedBlob)} | DefectCount: {int(r.DefectCount)} | Solidity: {r.Solidity:.3f}")
    else:
        logging.info("No misclassifications found on the test split.")


# ----------------- CLI and main -----------------
def main():
    parser = argparse.ArgumentParser(description="Optimized hand sign feature extractor + SVM classifier")
    parser.add_argument("data_folder", help="Path to dataset folder (class subfolders inside)")
    parser.add_argument("--no-visualize", action="store_true", help="Skip plotting/visualizations")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all images (ignore cache)")
    parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation")
    parser.add_argument("--no-save", action="store_true", help="Don't save trained model to disk")
    args = parser.parse_args()

    data_folder = Path(args.data_folder)

    start_all = time.perf_counter()
    df = analyze_features(data_folder, visualize=not args.no_visualize, force_recompute=args.force)
    logging.info(f"\nSample of extracted features (first 6 rows):\n{df.head(6)}")
    evaluate_svm(df, save_model=not args.no_save, do_cross_val=not args.no_cv)

    total_time = time.perf_counter() - start_all
    logging.info(f"All done. Total time: {total_time:.1f}s")


if __name__ == "__main__":
    data_folder = r"D:\Han\Minor\EVD3\RPSLS\Data"
    df = analyze_features(data_folder)
    visualize_features(df)
    evaluate_svm(df)


# Run from terminal: python optimized_hand_svm.py /path/to/Data
# To skip visualizations (headless server) add --no-visualize.
# To force reprocess all images (ignore cache): --force.
# The cache CSV (features_cache_svm_solidity_v2.csv) contains ImagePath, Hash, and computed features. Only changed/new images are reprocessed on subsequent runs.
# Trained model + encoder are saved to svm_model_joblib.pkl (a dict with pipeline, label_encoder, features).
