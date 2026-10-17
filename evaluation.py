"""
=============================================================================
KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING
Module: Performance Evaluation & Metrics Visualization
=============================================================================
This module computes standard statistical evaluation metrics on test datasets:
  1. Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score
  2. Multi-class Confusion Matrix with Seaborn Heatmap
  3. Per-Class Precision / Recall / F1 Breakdown & Classification Report
  4. Saving visual artifacts (confusion_matrix.png, classification_report.csv)

All values are strictly computed on real test splits. No hardcoded or fake metrics.
Note: For academic and research purposes only.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


DEFAULT_OUTPUT_DIR = "outputs"


def evaluate_model(
    model_or_pipeline: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Evaluates a trained model pipeline on a test feature set.

    Args:
        model_or_pipeline: Trained scikit-learn Pipeline or model bundle dict.
        X_test: Test features matrix.
        y_test: True test class labels.
        class_names: List of class names corresponding to index labels.

    Returns:
        Dict[str, Any]: Evaluation summary with metrics, confusion matrix, and reports.
    """
    # Extract pipeline if passed a model bundle dict
    if isinstance(model_or_pipeline, dict) and "pipeline" in model_or_pipeline:
        pipeline = model_or_pipeline["pipeline"]
    else:
        pipeline = model_or_pipeline

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Test feature set or test labels are empty.")

    # Model inference
    y_pred = pipeline.predict(X_test)

    # Calculate Core Performance Metrics (Weighted for robust multi-class representation)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    unique_labels = sorted(list(set(np.unique(y_test)).union(set(np.unique(y_pred)))))
    # Align target names for unique labels present
    eval_class_names = [
        class_names[i] if i < len(class_names) else f"Class_{i}" for i in unique_labels
    ]

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # Classification Report
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=unique_labels,
        target_names=eval_class_names,
        output_dict=True,
        zero_division=0
    )

    # Convert report to structured DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_test,
        "y_pred": y_pred,
        "class_names": eval_class_names,
        "unique_labels": unique_labels,
        "report_dict": report_dict,
        "report_df": report_df,
        "test_samples_count": len(y_test)
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders an publication-quality Seaborn heatmap of the confusion matrix.

    Args:
        cm: 2D confusion matrix array.
        class_names: List of class labels for ticks.
        title: Chart title.
        save_path: Optional path to save the generated image.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=120)

    # Normalize per row (True Class) for annotation percentage
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype(float), cm_sum, out=cm_perc, where=cm_sum != 0)

    # Create annotated labels (Count + Percentage)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=1.2,
        linecolor="#E2E8F0",
        square=True
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#1E293B")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold", labelpad=10, color="#334155")
    ax.set_ylabel("Actual / True Class", fontsize=11, fontweight="semibold", labelpad=10, color="#334155")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_per_class_metrics(
    report_dict: Dict[str, Any],
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a multi-bar chart comparing Precision, Recall, and F1-Score for each class.

    Args:
        report_dict: Dictionary returned by sklearn classification_report.
        class_names: List of target class names.
        save_path: Optional path to save figure.

    Returns:
        matplotlib.figure.Figure
    """
    valid_classes = [c for c in class_names if c in report_dict]
    if not valid_classes:
        valid_classes = [c for c in report_dict.keys() if c not in ["accuracy", "macro avg", "weighted avg"]]

    precisions = [report_dict[c]["precision"] * 100 for c in valid_classes]
    recalls = [report_dict[c]["recall"] * 100 for c in valid_classes]
    f1s = [report_dict[c]["f1-score"] * 100 for c in valid_classes]

    x = np.arange(len(valid_classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)

    rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#3B82F6", alpha=0.9)
    rects2 = ax.bar(x, recalls, width, label="Recall", color="#10B981", alpha=0.9)
    rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#8B5CF6", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="semibold", color="#334155")
    ax.set_title("Per-Class Metric Comparison", fontsize=13, fontweight="bold", pad=12, color="#1E293B")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_classes, fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Value labels on top of bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334155"
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def save_evaluation_outputs(
    eval_results: Dict[str, Any],
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> Dict[str, str]:
    """
    Saves confusion matrix image and classification report CSV to disk.

    Args:
        eval_results: Output dictionary from evaluate_model.
        output_dir: Directory to save generated artifacts.

    Returns:
        Dict[str, str]: Paths of generated files.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    csv_path = os.path.join(output_dir, "classification_report.csv")

    # Save Confusion Matrix
    cm = eval_results["confusion_matrix"]
    class_names = eval_results["class_names"]
    fig = plot_confusion_matrix(cm, class_names, save_path=cm_path)
    plt.close(fig)

    # Save Classification Report CSV
    report_df = eval_results["report_df"]
    report_df.to_csv(csv_path, index=True)

    return {
        "confusion_matrix_path": os.path.abspath(cm_path),
        "classification_report_path": os.path.abspath(csv_path)
    }
