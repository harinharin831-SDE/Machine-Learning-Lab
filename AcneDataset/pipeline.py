import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

from model import CLASSES, AcneClassifier, get_inference_transforms

# Try importing skimage, fallback gracefully if still installing
try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.measure import regionprops, label
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def load_image_and_metadata(image_input):
    """
    Step 1: Load input image and compute foundational image properties.
    Supports PIL Image, file path, or numpy array.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input)
    else:
        pil_img = image_input.convert('RGB')
        
    np_img = np.array(pil_img)
    h, w, c = np_img.shape
    
    # Compute metadata
    mean_rgb = np.mean(np_img, axis=(0, 1)).round(2)
    std_rgb = np.std(np_img, axis=(0, 1)).round(2)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # Shannon Entropy calculation (texture complexity)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    metadata = {
        'width': w,
        'height': h,
        'channels': c,
        'aspect_ratio': round(w / h, 3),
        'mean_rgb': mean_rgb.tolist(),
        'std_rgb': std_rgb.tolist(),
        'brightness': round(brightness, 2),
        'contrast': round(contrast, 2),
        'entropy': round(float(entropy), 3)
    }
    
    return pil_img, np_img, metadata


def preprocess_skin_lesion(np_img, clip_limit=2.0, blur_ksize=5, hair_removal=True):
    """
    Step 2: Preprocessing Pipeline for Skin Lesions
    - Hair removal (DullRazor algorithm simulation)
    - Median / Gaussian noise filtering
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Automated Lesion Mask Segmentation (Otsu thresholding + Morphological operations)
    """
    orig_rgb = np_img.copy()
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Hair Artifact Removal (Morphological Blackhat filtering)
    if hair_removal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        # Inpaint hair regions
        denoised = cv2.inpaint(orig_rgb, hair_mask, 3, cv2.INPAINT_TELEA)
    else:
        denoised = orig_rgb.copy()
        
    # 2. Gaussian Smoothing to suppress high-frequency skin noise
    if blur_ksize > 1 and blur_ksize % 2 == 1:
        denoised = cv2.GaussianBlur(denoised, (blur_ksize, blur_ksize), 0)
        
    # 3. CLAHE Contrast Enhancement in LAB color space (preserves natural skin color)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # 4. Lesion Segmentation Mask (Otsu on enhanced gray)
    enh_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    _, otsu_mask = cv2.threshold(enh_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean mask with morphological opening and closing
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, morph_kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, morph_kernel)
    
    # Segmented Lesion Foreground
    segmented_lesion = cv2.bitwise_and(enhanced_rgb, enhanced_rgb, mask=clean_mask)
    
    return {
        'original': orig_rgb,
        'denoised': denoised,
        'enhanced': enhanced_rgb,
        'lesion_mask': clean_mask,
        'segmented_lesion': segmented_lesion
    }


def extract_handcrafted_features(np_img, mask=None):
    """
    Step 3: Comprehensive Multi-Modal Feature Extraction
    - Color Moments (Mean, Std, Skewness) in RGB & HSV
    - Color Histograms (32-bin normalized)
    - GLCM Texture Descriptors (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM)
    - Local Binary Patterns (LBP) Texture Histogram
    - Morphological & Shape Metrics (ABCD rule: Area, Perimeter, Compactness, Asymmetry)
    """
    h, w, _ = np_img.shape
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    
    features = {}
    
    # A. Color Features
    for i, c_name in enumerate(['R', 'G', 'B']):
        channel = np_img[:, :, i]
        features[f'color_mean_{c_name}'] = float(np.mean(channel))
        features[f'color_std_{c_name}'] = float(np.std(channel))
        features[f'color_skew_{c_name}'] = float(np.mean(((channel - np.mean(channel)) / (np.std(channel) + 1e-5)) ** 3))
        
    for i, c_name in enumerate(['H', 'S', 'V']):
        channel = hsv[:, :, i]
        features[f'color_mean_{c_name}'] = float(np.mean(channel))
        features[f'color_std_{c_name}'] = float(np.std(channel))
        
    # B. Texture Features (GLCM)
    if HAS_SKIMAGE:
        # Downscale for fast GLCM
        small_gray = cv2.resize(gray, (128, 128))
        glcm = graycomatrix(small_gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            val = graycoprops(glcm, prop)
            features[f'glcm_{prop}'] = float(np.mean(val))
            
        # LBP (Local Binary Pattern)
        lbp = local_binary_pattern(small_gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        for i, val in enumerate(lbp_hist):
            features[f'lbp_bin_{i}'] = float(val)
    else:
        # Fallback OpenCV gradient texture features
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_magnitude_mean'] = float(np.mean(np.sqrt(gx**2 + gy**2)))
        features['gradient_magnitude_std'] = float(np.std(np.sqrt(gx**2 + gy**2)))
        features['laplacian_variance'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
    # C. Morphological & Shape Features
    if mask is not None and np.sum(mask) > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)
            perimeter = cv2.arcLength(largest_cnt, True)
            compactness = (perimeter ** 2) / (4 * np.pi * (area + 1e-5))
            
            features['lesion_area_px'] = float(area)
            features['lesion_perimeter_px'] = float(perimeter)
            features['lesion_compactness'] = float(compactness)
            features['lesion_area_ratio'] = float(area / (w * h))
        else:
            features['lesion_area_px'] = 0.0
            features['lesion_perimeter_px'] = 0.0
            features['lesion_compactness'] = 0.0
            features['lesion_area_ratio'] = 0.0
    else:
        features['lesion_area_px'] = 0.0
        features['lesion_perimeter_px'] = 0.0
        features['lesion_compactness'] = 0.0
        features['lesion_area_ratio'] = 0.0

    return features


def extract_dataset_features(data_dir, max_samples_per_class=50):
    """
    Extracts feature vectors for training classical ML models.
    """
    records = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            continue
            
        img_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        img_files = img_files[:max_samples_per_class]
        
        for p in img_files:
            try:
                pil_img = Image.open(p).convert('RGB')
                np_img = np.array(pil_img)
                prep = preprocess_skin_lesion(np_img)
                feats = extract_handcrafted_features(prep['enhanced'], prep['lesion_mask'])
                records.append(feats)
                labels.append(class_idx)
            except Exception as e:
                continue
                
    df_features = pd.DataFrame(records)
    return df_features, np.array(labels)


def train_ml_models(X_train, y_train, X_val=None, y_val=None):
    """
    Step 4: Machine Learning Classifier Training & Benchmarking
    Trains Random Forest, SVM (RBF), XGBoost / Gradient Boosting, KNN, and Logistic Regression.
    """
    models = {
        'Random Forest (100 Trees)': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        'Support Vector Machine (SVM RBF)': SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, random_state=42),
        'K-Nearest Neighbors (KNN k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Logistic Regression (L2)': LogisticRegression(max_iter=500, C=1.0, random_state=42)
    }
    
    results = {}
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        
        # Training metrics
        y_train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        val_acc = None
        val_f1 = None
        val_precision = None
        val_recall = None
        
        if X_val is not None and y_val is not None:
            y_val_pred = clf.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='weighted', zero_division=0)
            
        results[name] = {
            'model': clf,
            'train_acc': round(train_acc * 100, 2),
            'val_acc': round(val_acc * 100, 2) if val_acc is not None else None,
            'val_f1': round(val_f1 * 100, 2) if val_f1 is not None else None,
            'val_precision': round(val_precision * 100, 2) if val_precision is not None else None,
            'val_recall': round(val_recall * 100, 2) if val_recall is not None else None
        }
        
    return results
