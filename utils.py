"""
=============================================================================
KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING
Module: Dataset Management, Data Utilities & Demo Generator
=============================================================================
This module provides dataset loading, file discovery, validation, train/test
splitting, and synthetic ultrasound sample generation for testing the UI:
  1. Auto-detection of class folders in any directory structure
  2. Multi-extension support (.jpg, .jpeg, .png, .bmp, .tif, .tiff)
  3. Feature extraction pipeline execution across dataset
  4. Stratified Train/Test split with fallback for small sample sizes
  5. Synthetic Ultrasound dataset generation for instant demonstration

Note: For academic and research purposes only.
=============================================================================
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional, Callable
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_image
from feature_extraction import extract_features


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_dataset_summary(dataset_path: str) -> Dict[str, Any]:
    """
    Quickly scans the dataset directory without performing feature extraction.

    Args:
        dataset_path: Root folder containing class subdirectories.

    Returns:
        Dict[str, Any]: Scan summary with class names and file counts.
    """
    if not os.path.exists(dataset_path):
        return {
            "exists": False,
            "total_images": 0,
            "num_classes": 0,
            "class_names": [],
            "class_counts": {},
            "error": f"Path '{dataset_path}' does not exist."
        }

    # Find subdirectories
    subdirs = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".")
    ]

    class_counts = {}
    class_names = []
    total_images = 0

    for folder_name in sorted(subdirs):
        folder_path = os.path.join(dataset_path, folder_name)
        valid_files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS
        ]
        count = len(valid_files)
        if count > 0:
            class_names.append(folder_name)
            class_counts[folder_name] = count
            total_images += count

    return {
        "exists": True,
        "total_images": total_images,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_counts": class_counts,
        "error": None if len(class_names) >= 2 else (
            "Found less than 2 classes with valid images." if total_images > 0
            else "No valid images found in subdirectories."
        )
    }


def load_dataset(
    dataset_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_images_per_class: Optional[int] = None
) -> Dict[str, Any]:
    """
    Scans the dataset directory, applies preprocessing, and extracts HOG + GLCM
    features for every image.

    Args:
        dataset_path: Root directory containing class subdirectories.
        progress_callback: Optional callback function f(progress_ratio, status_text).
        max_images_per_class: Optional limit on images per class (useful for quick demo runs).

    Returns:
        Dict[str, Any]: Loaded dataset with X, y, class_names, file_paths, and stats.
    """
    summary = get_dataset_summary(dataset_path)
    if not summary["exists"]:
        raise FileNotFoundError(summary["error"])

    if summary["num_classes"] < 2:
        raise ValueError(
            f"Dataset at '{dataset_path}' has {summary['num_classes']} class(es). "
            "At least 2 distinct class folders with images are required for classification."
        )

    class_names = summary["class_names"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Collect all image file paths with their corresponding label
    image_tasks = []
    for class_name in class_names:
        folder_path = os.path.join(dataset_path, class_name)
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS
        ]
        files = sorted(files)
        if max_images_per_class and max_images_per_class > 0:
            files = files[:max_images_per_class]

        for file_path in files:
            image_tasks.append((file_path, class_to_idx[class_name], class_name))

    total_tasks = len(image_tasks)
    if total_tasks == 0:
        raise ValueError("No valid image files found in dataset.")

    features_list = []
    labels_list = []
    loaded_paths = []
    skipped_count = 0

    for i, (file_path, label_idx, class_name) in enumerate(image_tasks):
        if progress_callback:
            progress_ratio = float(i + 1) / total_tasks
            progress_callback(
                progress_ratio,
                f"Processing {i+1}/{total_tasks} [{class_name}]: {os.path.basename(file_path)}"
            )

        try:
            # Preprocess image
            prep = preprocess_image(file_path)
            # Extract HOG + GLCM features
            feat = extract_features(prep["enhanced"], return_breakdown=False)

            features_list.append(feat)
            labels_list.append(label_idx)
            loaded_paths.append(file_path)
        except Exception as e:
            skipped_count += 1
            print(f"Warning: Failed to process image '{file_path}': {e}")

    if len(features_list) == 0:
        raise RuntimeError("Failed to extract features from any images in the dataset.")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)

    # Calculate actual loaded counts per class
    actual_counts = {}
    for idx, name in enumerate(class_names):
        actual_counts[name] = int(np.sum(y == idx))

    return {
        "X": X,
        "y": y,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "file_paths": loaded_paths,
        "total_images": len(y),
        "class_counts": actual_counts,
        "n_features": X.shape[1],
        "skipped_images": skipped_count
    }


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Splits the dataset into training and testing sets with smart stratification.

    Args:
        X: Feature matrix.
        y: Class labels array.
        test_size: Fraction of samples reserved for testing (default: 0.20).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, is_stratified)
    """
    # Verify minimum samples per class for stratification
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique) >= 2 and all(c >= 2 for c in counts)

    stratify_arg = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    return X_train, X_test, y_train, y_test, can_stratify


def create_sample_synthetic_dataset(
    output_dir: str = "dataset",
    images_per_class: int = 15,
    image_size: Tuple[int, int] = (128, 128)
) -> Dict[str, int]:
    """
    Generates a realistic synthetic ultrasound dataset with 4 categories for demo testing:
      - Normal: Smooth homogeneous renal parenchyma with central echogenic sinus.
      - Cyst: Anechoic dark circular lesion with smooth margins and distal acoustic enhancement.
      - Stone: Bright hyperechoic focus with a dark distal acoustic shadowing band.
      - Tumor: Heterogeneous irregular mass with mixed hyper/hypoechoic tissue regions.

    Args:
        output_dir: Root dataset folder to create.
        images_per_class: Number of synthetic images to generate per class.
        image_size: Image dimensions (width, height).

    Returns:
        Dict[str, int]: Created classes and count of generated images.
    """
    categories = ["Normal", "Cyst", "Stone", "Tumor"]
    created_stats = {}
    np.random.seed(42)

    width, height = image_size

    for category in categories:
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        for img_idx in range(1, images_per_class + 1):
            # 1. Base Ultrasound Background (Renal shape + Speckle Noise)
            # Create kidney elliptical boundary
            img = np.zeros((height, width), dtype=np.float32)
            center_x, center_y = width // 2, height // 2
            axes_x, axes_y = int(width * 0.42), int(height * 0.35)

            y_grid, x_grid = np.ogrid[:height, :width]
            dist_from_center = (
                ((x_grid - center_x) / axes_x) ** 2 + ((y_grid - center_y) / axes_y) ** 2
            )
            kidney_mask = dist_from_center <= 1.0

            # Parenchyma background tissue (moderate gray level ~70-90)
            img[kidney_mask] = 80.0

            # Central Renal Sinus (bright echogenic core)
            sinus_mask = (
                ((x_grid - center_x) / (axes_x * 0.45)) ** 2
                + ((y_grid - center_y) / (axes_y * 0.4)) ** 2
            ) <= 1.0
            img[sinus_mask] = 160.0

            # Add characteristic Rayleigh / Speckle noise
            speckle = np.random.rayleigh(scale=18.0, size=(height, width))
            img = img + speckle

            # 2. Add Pathology Specific Morphological Signatures
            if category == "Cyst":
                # Dark anechoic fluid-filled circular cyst
                cx = center_x + np.random.randint(-15, 15)
                cy = center_y + np.random.randint(-15, 15)
                cradius = np.random.randint(14, 22)
                cyst_dist = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
                cyst_mask = cyst_dist <= (cradius ** 2)
                # Anechoic interior (dark ~15)
                img[cyst_mask] = np.random.normal(18, 5, size=img[cyst_mask].shape)

                # Posterior acoustic enhancement (bright band below cyst)
                enhancement_mask = (cyst_dist > cradius ** 2) & (
                    np.abs(x_grid - cx) < cradius * 0.9
                ) & (y_grid > cy + cradius) & (y_grid < cy + cradius + 28)
                img[enhancement_mask] += 45.0

            elif category == "Stone":
                # Bright hyperechoic calculus (kidney stone)
                sx = center_x + np.random.randint(-18, 18)
                sy = center_y + np.random.randint(-18, 10)
                sradius = np.random.randint(5, 9)
                stone_mask = ((x_grid - sx) ** 2 + (y_grid - sy) ** 2) <= (sradius ** 2)
                img[stone_mask] = np.random.normal(240, 10, size=img[stone_mask].shape)

                # Acoustic Shadowing: dark band extending downward
                shadow_mask = (
                    (np.abs(x_grid - sx) <= sradius + 2)
                    & (y_grid > sy + sradius)
                    & (y_grid < height - 5)
                )
                img[shadow_mask] = np.random.normal(20, 6, size=img[shadow_mask].shape)

            elif category == "Tumor":
                # Irregular heterogeneous tissue mass
                tx = center_x + np.random.randint(-12, 12)
                ty = center_y + np.random.randint(-12, 12)
                tradius = np.random.randint(18, 26)

                # Perturbed boundary
                angle = np.arctan2(y_grid - ty, x_grid - tx)
                deformed_radius = tradius * (1.0 + 0.25 * np.sin(3 * angle) + 0.15 * np.cos(5 * angle))
                tumor_dist = np.sqrt((x_grid - tx) ** 2 + (y_grid - ty) ** 2)
                tumor_mask = tumor_dist <= deformed_radius

                # Heterogeneous texture (mixed bright and dark nodules)
                tumor_texture = np.random.normal(120, 40, size=img[tumor_mask].shape)
                img[tumor_mask] = tumor_texture

            # Clamp and convert to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)

            # Apply light ultrasound smoothing
            img = cv2.GaussianBlur(img, (3, 3), 0)

            # Save as PNG image
            file_name = f"{category.lower()}_sample_{img_idx:03d}.png"
            file_path = os.path.join(cat_dir, file_name)
            Image.fromarray(img).save(file_path)

        created_stats[category] = images_per_class

    return created_stats
