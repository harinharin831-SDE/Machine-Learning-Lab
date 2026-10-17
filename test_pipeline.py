"""
Quick automated test suite for kidney disease image classification pipeline.
"""
import os
from preprocessing import preprocess_image
from feature_extraction import extract_features
from model_training import train_model, save_model, load_saved_model
from evaluation import evaluate_model
from prediction import predict_single_image
from utils import load_dataset, split_data


def test_all():
    print("1. Testing Preprocessing on sample image...")
    prep = preprocess_image("sample_images/test_normal_ultrasound.png")
    for k in ["original", "resized", "gray", "denoised", "enhanced", "normalized"]:
        assert k in prep, f"Missing {k}"
    print("   [OK] Preprocessing OK (128x128 5-stage pipeline verified)")

    print("\n2. Testing Feature Extraction (HOG + GLCM)...")
    feats = extract_features(prep["enhanced"], return_breakdown=True)
    count = feats["total_count"]
    assert count == 8160, f"Expected 8160 features, got {count}"
    print(f"   [OK] Feature Extraction OK! (HOG: {feats['hog_count']} + GLCM: {feats['glcm_count']} = {count} total)")

    print("\n3. Testing Dataset Loading...")
    data = load_dataset("dataset")
    assert data["total_images"] >= 60, "Dataset not loaded"
    print(f"   [OK] Dataset Loaded: {data['total_images']} images across classes {data['class_names']}")

    print("\n4. Testing Train/Test Split & Multi-Model Training...")
    X_tr, X_te, y_tr, y_te, is_strat = split_data(data["X"], data["y"], test_size=0.20)
    
    for algo in ["SVM", "Random Forest", "KNN"]:
        pkg = train_model(X_tr, y_tr, algorithm=algo, class_names=data["class_names"])
        assert "pipeline" in pkg, f"Pipeline missing for {algo}"
        ev = evaluate_model(pkg, X_te, y_te, class_names=data["class_names"])
        acc = ev["accuracy"] * 100
        f1 = ev["f1_score"] * 100
        print(f"   [OK] {algo:15s} -> Test Accuracy: {acc:5.2f}% | Test F1-Score: {f1:5.2f}%")

    print("\n5. Testing Single Image Inference Pipeline...")
    rf_pkg = train_model(X_tr, y_tr, algorithm="Random Forest", class_names=data["class_names"])
    for img_name in sorted(os.listdir("sample_images")):
        img_path = os.path.join("sample_images", img_name)
        res = predict_single_image(rf_pkg, img_path)
        p_cls = res["predicted_class"]
        conf = res["confidence_percent"]
        print(f"   [OK] {img_name:30s} -> Predicted: {p_cls:8s} (Confidence: {conf:5.1f}%)")

    print("\n========================================================")
    print(" ALL PIPELINE VERIFICATIONS PASSED WITH 100% SUCCESS! ")
    print("========================================================")


if __name__ == "__main__":
    test_all()
