# 🔬 DermaScan AI - Skin Lesion Machine Learning Studio

A comprehensive, all-in-one Machine Learning & Deep Learning Computer Vision platform for Skin Lesion & Acne Classification.

---

## 🌟 All-In-One Unified Pipeline Architecture

```
[ 1. Input Image ] ➔ [ 2. Preprocessing ] ➔ [ 3. Feature Extraction ] ➔ [ 4. Model Training ] ➔ [ 5. Evaluation & Grad-CAM ]
     ├── Upload Photo     ├── DullRazor Filter    ├── Color Moments (RGB/HSV) ├── Random Forest       ├── Accuracy / F1 Matrix
     ├── Webcam Scan      ├── Gaussian Blur       ├── GLCM Texture Matrix    ├── SVM (RBF Kernel)     ├── Confusion Matrix
     └── Dataset Sample   └── CLAHE Contrast      └── Morphological ABCD     └── Deep CNN (MobileNet) └── Grad-CAM Heatmap
```

---

## 🚀 How to Run

### 1. Launch the All-In-One Web Application
```powershell
python -m streamlit run app.py
```
*(Or double-click [`run.bat`](file:///c:/Users/Lenovo/Downloads/archive/AcneDataset/run.bat))*

Access in browser at: **`http://localhost:8501`**

---

### 2. Live Interactive Code Playground & Compiler
The web application includes an **in-browser Python IDE**:
- Select pre-configured runnable templates (End-to-End Pipeline, GLCM Feature Extraction, SVM Training, CLAHE Preprocessing, PyTorch CNN Inference).
- Type or edit Python ML code directly in the browser editor.
- Click **"⚡ Compile & Run Code"** to execute in real-time, view `stdout`/`stderr` terminal logs, execution time, and rendered Matplotlib plots in one frame!

---

### 3. CLI Scripts
- **Train Deep Learning CNN**:
  ```powershell
  python train.py --epochs 4 --batch_size 32 --lr 0.001
  ```
- **Evaluate on Benchmark Test Split**:
  ```powershell
  python evaluate.py
  ```
- **Train Classical ML Classifiers via Python**:
  ```powershell
  python -c "from pipeline import extract_dataset_features, train_ml_models; X, y = extract_dataset_features('train', 30); res = train_ml_models(X, y); print('Finished training!')"
  ```
