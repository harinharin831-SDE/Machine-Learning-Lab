"""
=============================================================================
KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING
Module: Feature Extraction (HOG + GLCM)
=============================================================================
This module extracts structural, edge-gradient, and textural patterns from
preprocessed ultrasound kidney images:
  1. HOG (Histogram of Oriented Gradients):
     - Captures edge directions, structural gradients, and boundary shapes.
     - Parameters: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
                   block_norm='L2-Hys'.
  2. GLCM (Gray-Level Co-occurrence Matrix):
     - Quantifies spatial pixel-intensity correlations & micro-textures.
     - Properties: Contrast, Dissimilarity, Homogeneity, Energy, Correlation.
     - Multi-scale (distances=[1, 2, 3]) and multi-directional (0°, 45°, 90°, 135°).

Combined Feature Vector = [ HOG Features | GLCM Features ]
Note: For academic and research purposes only.
=============================================================================
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from skimage.feature import hog, graycomatrix, graycoprops
import cv2


# Standard HOG configuration as per requirements
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# Standard GLCM configuration
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 degrees
GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def extract_hog_features(
    image_gray: np.ndarray,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell: Tuple[int, int] = HOG_PIXELS_PER_CELL,
    cells_per_block: Tuple[int, int] = HOG_CELLS_PER_BLOCK,
    block_norm: str = HOG_BLOCK_NORM,
    visualize: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extracts Histogram of Oriented Gradients (HOG) descriptor vector from a grayscale image.

    Args:
        image_gray: 2D Grayscale uint8 array (128x128).
        orientations: Number of gradient orientation bins (default: 9).
        pixels_per_cell: Spatial cell dimensions in pixels (default: 8x8).
        cells_per_block: Number of cells in each normalization block (default: 2x2).
        block_norm: Block normalization method (default: 'L2-Hys').
        visualize: Whether to compute and return the 2D HOG visualization image.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - 1D float array of HOG feature values (8100 elements for 128x128).
            - 2D HOG visualization image array (if visualize=True, else None).
    """
    if len(image_gray.shape) != 2:
        if len(image_gray.shape) == 3:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Expected 2D grayscale array, got shape {image_gray.shape}")

    # Ensure image is uint8 in [0, 255]
    if image_gray.dtype != np.uint8:
        if image_gray.max() <= 1.0:
            image_gray = (image_gray * 255).astype(np.uint8)
        else:
            image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)

    if visualize:
        features, hog_image = hog(
            image_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=True,
            transform_sqrt=True
        )
        return features.astype(np.float32), hog_image
    else:
        features = hog(
            image_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=False,
            transform_sqrt=True
        )
        return features.astype(np.float32), None


def extract_glcm_features(
    image_gray: np.ndarray,
    distances: List[int] = None,
    angles: List[float] = None,
    properties: List[str] = None
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Calculates Gray-Level Co-occurrence Matrix (GLCM) statistical texture descriptors.

    Args:
        image_gray: 2D Grayscale uint8 array.
        distances: List of pixel separation distances (default: [1, 2, 3]).
        angles: List of pixel pair angles in radians (default: [0, 45°, 90°, 135°]).
        properties: Statistical properties to compute (default: contrast, dissimilarity,
                    homogeneity, energy, correlation).

    Returns:
        Tuple containing:
          1. 1D float array of flattened GLCM features across all properties (len = properties * distances * angles).
          2. Dict of mean values for each property (useful for UI dashboard).
          3. Dict of raw 2D arrays (len(distances) x len(angles)) for each property.
    """
    if distances is None:
        distances = GLCM_DISTANCES
    if angles is None:
        angles = GLCM_ANGLES
    if properties is None:
        properties = GLCM_PROPERTIES

    if len(image_gray.shape) != 2:
        if len(image_gray.shape) == 3:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Expected 2D grayscale array, got shape {image_gray.shape}")

    # Ensure uint8
    if image_gray.dtype != np.uint8:
        if image_gray.max() <= 1.0:
            image_gray = (image_gray * 255).astype(np.uint8)
        else:
            image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)

    # Compute GLCM matrix with 256 gray levels
    glcm = graycomatrix(
        image_gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    feature_list = []
    summary_metrics = {}
    detailed_matrices = {}

    for prop in properties:
        val_matrix = graycoprops(glcm, prop)  # shape: (len(distances), len(angles))
        detailed_matrices[prop] = val_matrix
        feature_list.extend(val_matrix.flatten())
        summary_metrics[prop] = float(np.mean(val_matrix))

    glcm_vector = np.array(feature_list, dtype=np.float32)
    return glcm_vector, summary_metrics, detailed_matrices


def extract_features(
    image: np.ndarray,
    return_breakdown: bool = False
) -> Any:
    """
    Extracts the combined (HOG + GLCM) numerical feature vector from an image.

    Args:
        image: Grayscale or RGB image array (128x128).
        return_breakdown: If True, returns a comprehensive dictionary with HOG and GLCM
                          intermediates, metrics, visualization image, and vector slices.
                          If False, returns only the combined 1D feature vector.

    Returns:
        np.ndarray (if return_breakdown=False) or Dict[str, Any] (if return_breakdown=True).
    """
    # Ensure grayscale 2D array
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image.copy()

    # 1. HOG features
    hog_features, hog_vis = extract_hog_features(image_gray, visualize=return_breakdown)

    # 2. GLCM features
    glcm_features, glcm_summary, glcm_matrices = extract_glcm_features(image_gray)

    # 3. Concatenate feature vectors
    combined_vector = np.hstack((hog_features, glcm_features)).astype(np.float32)

    if not return_breakdown:
        return combined_vector

    return {
        "feature_vector": combined_vector,
        "hog_vector": hog_features,
        "hog_image": hog_vis,
        "glcm_vector": glcm_features,
        "glcm_summary": glcm_summary,
        "glcm_matrices": glcm_matrices,
        "total_count": len(combined_vector),
        "hog_count": len(hog_features),
        "glcm_count": len(glcm_features),
        "first_20": combined_vector[:20].tolist(),
        "shape": combined_vector.shape
    }
