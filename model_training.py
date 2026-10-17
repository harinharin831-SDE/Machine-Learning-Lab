"""
=============================================================================
KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING
Module: Machine Learning Model Training & Persistence
=============================================================================
This module handles training, hyperparameter management, pipeline construction,
and model persistence using Scikit-Learn and Joblib:
  1. Support Vector Machine (SVM / SVC) with RBF kernel and probability calibration
  2. Random Forest Classifier (Ensemble of Decision Trees)
  3. K-Nearest Neighbors (KNN) Classifier
  4. Scikit-learn Pipeline with StandardScaler
  5. Joblib serialization and validation

Note: For academic and research purposes only.
=============================================================================
"""

import os
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


SUPPORTED_ALGORITHMS = ["SVM", "Random Forest", "KNN"]
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "kidney_model.pkl")


def create_classifier(algorithm: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Instantiates the requested scikit-learn classifier with optimized default or custom hyperparameters.

    Args:
        algorithm: One of 'SVM', 'Random Forest', or 'KNN'.
        params: Optional dictionary of hyperparameter overrides.

    Returns:
        Classifier instance.
    """
    params = params or {}
    algo_upper = algorithm.strip().upper()

    if algo_upper in ["SVM", "SUPPORT VECTOR MACHINE", "SVC"]:
        c_val = params.get("C", 1.0)
        kernel = params.get("kernel", "rbf")
        gamma = params.get("gamma", "scale")
        return SVC(
            C=c_val,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42
        )

    elif algo_upper in ["RANDOM FOREST", "RF", "RANDOMFOREST"]:
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", None)
        min_samples_split = params.get("min_samples_split", 2)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

    elif algo_upper in ["KNN", "K-NEAREST NEIGHBORS", "KNEIGHBORS"]:
        n_neighbors = params.get("n_neighbors", 5)
        weights = params.get("weights", "uniform")
        metric = params.get("metric", "minkowski")
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )

    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from {SUPPORTED_ALGORITHMS}")


def build_pipeline(algorithm: str, params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Constructs a complete ML pipeline: StandardScaler -> Classifier.

    Args:
        algorithm: Classifier name ('SVM', 'Random Forest', 'KNN').
        params: Optional hyperparameter dict.

    Returns:
        sklearn.pipeline.Pipeline
    """
    clf = create_classifier(algorithm, params)
    
    # Note: StandardScaler is critical for distance-based SVM and KNN;
    # it is also beneficial/neutral for tree ensembles in high-dimensional feature spaces.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    return pipeline


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    algorithm: str = "SVM",
    params: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Trains the selected ML model on the feature matrix and returns the packaged model bundle.

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Training label array of shape (n_samples,).
        algorithm: Algorithm identifier ('SVM', 'Random Forest', 'KNN').
        params: Optional dictionary of hyperparameters.
        class_names: List of human-readable class names matching label indices.

    Returns:
        Dict[str, Any]: Model package ready for saving or evaluation.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training dataset cannot be empty.")

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Training requires at least 2 distinct classes. Found only: {unique_classes}"
        )

    # Build Pipeline
    pipeline = build_pipeline(algorithm, params)

    # Fit Pipeline
    pipeline.fit(X_train, y_train)

    # Class names mapping
    if class_names is None:
        class_names = [f"Class_{c}" for c in sorted(unique_classes)]

    model_package = {
        "pipeline": pipeline,
        "algorithm": algorithm,
        "hyperparameters": params or {},
        "class_names": class_names,
        "n_classes": len(class_names),
        "n_features": X_train.shape[1],
        "image_size": (128, 128),
        "feature_config": {
            "method": "HOG + GLCM",
            "hog_orientations": 9,
            "hog_pixels_per_cell": (8, 8),
            "hog_cells_per_block": (2, 2),
            "glcm_properties": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples_count": len(X_train)
    }

    return model_package


def save_model(
    model_package: Dict[str, Any],
    filepath: str = DEFAULT_MODEL_PATH
) -> str:
    """
    Serializes and saves the model package to disk using Joblib.

    Args:
        model_package: Model bundle containing pipeline, metadata, and class mapping.
        filepath: Target destination file path.

    Returns:
        str: Absolute path of saved model file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(model_package, filepath)
    return os.path.abspath(filepath)


def load_saved_model(filepath: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Loads and validates a trained model package from disk.

    Args:
        filepath: Path to the .pkl model file.

    Returns:
        Dict[str, Any]: Loaded model package.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    package = joblib.load(filepath)
    if not isinstance(package, dict) or "pipeline" not in package:
        # Legacy/direct pipeline fallback
        if hasattr(package, "predict"):
            package = {
                "pipeline": package,
                "algorithm": "Unknown",
                "class_names": ["Normal", "Abnormal"],
                "n_classes": 2,
                "feature_config": {"method": "HOG + GLCM"}
            }
        else:
            raise ValueError("Invalid model file format.")

    return package
