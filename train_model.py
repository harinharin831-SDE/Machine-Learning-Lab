"""
=============================================================================
KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING
Script: Standalone Model Training & Evaluation CLI
=============================================================================
This script provides an automated command-line workflow:
  1. Loads ultrasound images from dataset directory
  2. Applies 5-stage preprocessing & HOG + GLCM feature extraction
  3. Performs stratified Train/Test split
  4. Trains the selected ML algorithm (SVM, Random Forest, or KNN)
  5. Evaluates test set performance (Accuracy, Precision, Recall, F1-Score)
  6. Saves trained model to 'models/kidney_model.pkl'
  7. Generates 'outputs/confusion_matrix.png' and 'outputs/classification_report.csv'

Usage:
  python train_model.py --algorithm SVM
  python train_model.py --algorithm "Random Forest" --dataset dataset
  python train_model.py --generate-demo-data

Note: For academic and research purposes only. Not for medical diagnosis.
=============================================================================
"""

import argparse
import os
import sys

from utils import load_dataset, split_data, create_sample_synthetic_dataset, get_dataset_summary
from model_training import train_model, save_model, SUPPORTED_ALGORITHMS
from evaluation import evaluate_model, save_evaluation_outputs, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Train Machine Learning models for Kidney Disease Image Classification"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path to dataset directory containing class subfolders (default: dataset)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SVM",
        choices=["SVM", "Random Forest", "KNN"],
        help="Machine Learning algorithm to train (default: SVM)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data reserved for testing (default: 0.20)"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/kidney_model.pkl",
        help="Output path for the trained model bundle (default: models/kidney_model.pkl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save evaluation charts and reports (default: outputs)"
    )
    parser.add_argument(
        "--generate-demo-data",
        action="store_true",
        help="Generate synthetic demo ultrasound dataset if dataset directory is missing"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING ")
    print(" Academic & Educational ML Mini-Project ")
    print("=" * 70)
    print(f"Algorithm: {args.algorithm}")
    print(f"Dataset Path: {args.dataset}")
    print("-" * 70)

    # 1. Check Dataset
    summary = get_dataset_summary(args.dataset)
    if not summary["exists"] or summary["total_images"] == 0 or summary["num_classes"] < 2:
        if args.generate_demo_data or not os.path.exists(args.dataset):
            print("[INFO] Dataset not found or insufficient classes. Generating synthetic demo dataset...")
            created = create_sample_synthetic_dataset(output_dir=args.dataset, images_per_class=15)
            print(f"[SUCCESS] Created demo dataset with {sum(created.values())} images across classes: {list(created.keys())}")
        else:
            print(f"[ERROR] Invalid dataset at '{args.dataset}': {summary['error']}")
            print("Please place class folders (e.g. Normal, Cyst, Stone, Tumor) inside the dataset directory,")
            print("or run with --generate-demo-data flag.")
            sys.exit(1)

    # 2. Load Dataset & Extract Features
    print("\n[STEP 1/5] Loading images & extracting HOG + GLCM features...")
    def print_progress(ratio, text):
        bar_len = 25
        filled = int(round(bar_len * ratio))
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {ratio*100:5.1f}% - {text[:40]}")
        sys.stdout.flush()

    try:
        data = load_dataset(args.dataset, progress_callback=print_progress)
        print()
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    X = data["X"]
    y = data["y"]
    class_names = data["class_names"]

    print(f"[INFO] Loaded {len(y)} total images across {len(class_names)} classes.")
    for c_name, count in data["class_counts"].items():
        print(f"  - {c_name}: {count} images")
    print(f"[INFO] Feature vector shape per image: {X.shape[1]} (HOG: 8100 + GLCM: 60)")

    # 3. Train/Test Split
    print("\n[STEP 2/5] Performing Train/Test Split...")
    X_train, X_test, y_train, y_test, is_stratified = split_data(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"[INFO] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    print(f"[INFO] Stratified split applied: {is_stratified}")

    # 4. Train Model
    print(f"\n[STEP 3/5] Training {args.algorithm} classifier with StandardScaler pipeline...")
    model_package = train_model(
        X_train=X_train,
        y_train=y_train,
        algorithm=args.algorithm,
        class_names=class_names
    )
    print(f"[SUCCESS] Model training completed successfully.")

    # 5. Evaluate Model
    print("\n[STEP 4/5] Evaluating performance on test set...")
    eval_results = evaluate_model(
        model_or_pipeline=model_package,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names
    )

    acc = eval_results["accuracy"] * 100
    prec = eval_results["precision"] * 100
    rec = eval_results["recall"] * 100
    f1 = eval_results["f1_score"] * 100

    print("\n" + "=" * 45)
    print("        TEST EVALUATION RESULTS")
    print("=" * 45)
    print(f"  Accuracy  : {acc:6.2f}%")
    print(f"  Precision : {prec:6.2f}%")
    print(f"  Recall    : {rec:6.2f}%")
    print(f"  F1 Score  : {f1:6.2f}%")
    print("=" * 45)

    print("\nClassification Report:")
    print(eval_results["report_df"].to_string())

    # Attach evaluation metrics to model package
    model_package["test_metrics"] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "test_samples": len(y_test)
    }

    # 6. Save Model and Artifacts
    print("\n[STEP 5/5] Saving model and output artifacts...")
    saved_model_path = save_model(model_package, filepath=args.output_model)
    print(f"[SAVED] Trained model: {saved_model_path}")

    saved_outputs = save_evaluation_outputs(eval_results, output_dir=args.output_dir)
    print(f"[SAVED] Confusion Matrix heatmap: {saved_outputs['confusion_matrix_path']}")
    print(f"[SAVED] Classification Report CSV: {saved_outputs['classification_report_path']}")

    print("\n" + "=" * 70)
    print(" TRAINING PIPELINE FINISHED SUCCESSFULLY")
    print(" Run web application with: python -m streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
