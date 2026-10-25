import os
import io
import sys
import glob
import json
import random
import time
import contextlib
import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Safe optional Plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from model import (
    AcneClassifier, 
    get_inference_transforms, 
    compute_gradcam, 
    overlay_gradcam, 
    CLASSES, 
    ACNE_KNOWLEDGE_BASE
)
from pipeline import (
    load_image_and_metadata,
    preprocess_skin_lesion,
    extract_handcrafted_features,
    extract_dataset_features,
    train_ml_models
)

# Set page config
st.set_page_config(
    page_title="DermaScan AI - Skin Lesion Machine Learning Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre, .terminal-box {
        font-family: 'Fira Code', monospace !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 118, 110, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .step-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #0f766e;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .diagnosis-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1rem;
    }
    
    .terminal-box {
        background: #0f172a;
        color: #38bdf8;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #334155;
        font-size: 0.88rem;
        overflow-x: auto;
        white-space: pre-wrap;
        line-height: 1.5;
    }
    
    .ingredient-tag {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 5px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
        display: inline-block;
    }
    
    .routine-box {
        border-left: 4px solid #0d9488;
        background: #f0fdfa;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_deep_model(weights_path="acne_classifier.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AcneClassifier(num_classes=len(CLASSES), pretrained=False)
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            val_acc = checkpoint.get('val_acc', None)
        else:
            model.load_state_dict(checkpoint)
            val_acc = None
        loaded = True
    else:
        model = AcneClassifier(num_classes=len(CLASSES), pretrained=True)
        val_acc = None
        loaded = False
        
    model.to(device)
    model.eval()
    return model, device, loaded, val_acc


def predict_deep_image(model, device, pil_image):
    transform = get_inference_transforms()
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    inference_time = (time.time() - t0) * 1000
    
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASSES[predicted_idx]
    confidence = float(probabilities[predicted_idx])
    
    # Compute Grad-CAM
    cam_map, _ = compute_gradcam(model, img_tensor, target_class=predicted_idx)
    overlay_img, heatmap_img = overlay_gradcam(pil_image, cam_map)
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'probabilities': {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))},
        'inference_time_ms': inference_time,
        'overlay_img': overlay_img,
        'heatmap_img': heatmap_img
    }


# Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;">🔬 DermaScan AI • ML & Deep Learning Studio</h1>
            <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 1rem;">
                Unified Skin Lesion Classification: Image Loading ➔ Preprocessing ➔ Feature Extraction ➔ Multi-Model Training ➔ Performance Evaluation ➔ Live Code Playground
            </p>
        </div>
        <div style="text-align: right; margin-top: 8px;">
            <span style="background: rgba(255,255,255,0.15); padding: 5px 12px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">
                ⚡ Classical ML (SVM, RF, KNN) + Deep CNN
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load deep model
model, device, weights_loaded, val_acc = load_deep_model()

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.markdown("### ⚙️ Pipeline Navigation")
    st.info("This application consolidates the entire machine learning lifecycle into an interactive, all-in-one platform.")
    
    st.markdown("#### 📦 ML Frameworks Active:")
    st.markdown("• **PyTorch & Torchvision** (CNN Inference & Grad-CAM)\n• **Scikit-Learn** (SVM, Random Forest, KNN)\n• **OpenCV & NumPy** (Image Preprocessing & Filters)\n• **Matplotlib & Plotly** (Metrics & Heatmaps)")
    
    st.divider()
    st.markdown("### 📋 Supported Acne Categories")
    for c in CLASSES:
        info = ACNE_KNOWLEDGE_BASE[c]
        st.markdown(f"• **{c}** (`{info['medical_name']}`)")
        
    st.divider()
    st.caption("⚕️ **Educational & Research System**: DermaScan AI provides machine learning diagnostics and explainable feature visualizations.")


# Main Application Navigation Tabs
tab_pipeline, tab_features, tab_ml_arena, tab_code_playground, tab_docs = st.tabs([
    "🚀 End-to-End Pipeline (Image ➔ Preprocessing ➔ Features ➔ Diagnosis)",
    "🧬 Step-by-Step Feature Extractor & Morphology Lab",
    "🤖 Multi-Model ML Training & Evaluation Arena",
    "💻 Live Python Code Playground & Compiler Console",
    "📖 CLI Compilation & Execution Guide"
])


# ==============================================================================
# TAB 1: Complete End-to-End Pipeline
# ==============================================================================
with tab_pipeline:
    st.markdown("### 🎯 Full 5-Stage Machine Learning Pipeline in One View")
    
    # -------------------------------------------------------------
    # STAGE 1: LOAD INPUT IMAGE
    # -------------------------------------------------------------
    st.markdown('<span class="step-badge">STAGE 1: Image Acquisition & Metadata</span>', unsafe_allow_html=True)
    
    c_in1, c_in2 = st.columns([1, 1.2], gap="medium")
    
    with c_in1:
        input_choice = st.radio(
            "Select Input Source:",
            ["📁 Upload Custom Skin Photo", "📷 Webcam Live Capture", "🖼️ Curated Benchmark Dataset Sample"],
            horizontal=True
        )
        
        raw_pil = None
        ground_truth_label = None
        
        if input_choice == "📁 Upload Custom Skin Photo":
            uploaded = st.file_uploader("Upload Skin Lesion Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                raw_pil = Image.open(uploaded).convert('RGB')
        elif input_choice == "📷 Webcam Live Capture":
            cam_shot = st.camera_input("Take Live Photo")
            if cam_shot:
                raw_pil = Image.open(cam_shot).convert('RGB')
        elif input_choice == "🖼️ Curated Benchmark Dataset Sample":
            sel_class = st.selectbox("Select Class to Sample:", CLASSES, index=0)
            class_folder = os.path.join("test", sel_class)
            if os.path.exists(class_folder):
                samples = glob.glob(os.path.join(class_folder, "*.*"))
                if samples:
                    sel_path = st.selectbox("Select Sample Image:", samples[:25], format_func=lambda x: os.path.basename(x))
                    if sel_path:
                        raw_pil = Image.open(sel_path).convert('RGB')
                        ground_truth_label = sel_class
                        
    with c_in2:
        if raw_pil is not None:
            pil_img, np_img, meta = load_image_and_metadata(raw_pil)
            st.image(pil_img, caption="Acquired Input Image" + (f" (Ground Truth: {ground_truth_label})" if ground_truth_label else ""), use_container_width=True)
            
            # Metadata metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Dimensions", f"{meta['width']} x {meta['height']}")
            with col_m2:
                st.metric("Brightness", f"{meta['brightness']}")
            with col_m3:
                st.metric("Contrast", f"{meta['contrast']}")
            with col_m4:
                st.metric("Shannon Entropy", f"{meta['entropy']}")
        else:
            st.info("👈 Please select or upload an image to begin pipeline execution.")

    if raw_pil is not None:
        st.divider()
        # -------------------------------------------------------------
        # STAGE 2: PREPROCESSING & LESION SEGMENTATION
        # -------------------------------------------------------------
        st.markdown('<span class="step-badge">STAGE 2: Preprocessing & Lesion Segmentation</span>', unsafe_allow_html=True)
        st.caption("Interactive Preprocessing controls: CLAHE Contrast Enhancement, Gaussian Noise Reduction, and Otsu Morphological Masking.")
        
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            clahe_clip = st.slider("CLAHE Clip Limit (Contrast):", min_value=1.0, max_value=5.0, value=2.5, step=0.5)
        with p_c2:
            blur_size = st.slider("Gaussian Blur Kernel Size:", min_value=1, max_value=11, value=5, step=2)
        with p_c3:
            hair_rem = st.checkbox("Apply DullRazor Hair Artifact Filter", value=True)
            
        prep_res = preprocess_skin_lesion(np_img, clip_limit=clahe_clip, blur_ksize=blur_size, hair_removal=hair_rem)
        
        st_cols = st.columns(4)
        with st_cols[0]:
            st.image(prep_res['original'], caption="1. Raw RGB Image", use_container_width=True)
        with st_cols[1]:
            st.image(prep_res['denoised'], caption="2. Denoised & Hair-Filtered", use_container_width=True)
        with st_cols[2]:
            st.image(prep_res['enhanced'], caption="3. CLAHE Contrast Enhanced", use_container_width=True)
        with st_cols[3]:
            st.image(prep_res['segmented_lesion'], caption="4. Segmented Lesion ROI", use_container_width=True)

        st.divider()
        # -------------------------------------------------------------
        # STAGE 3: FEATURE EXTRACTION
        # -------------------------------------------------------------
        st.markdown('<span class="step-badge">STAGE 3: Multi-Modal Feature Extraction</span>', unsafe_allow_html=True)
        st.caption("Extracted numerical feature descriptors combining Color Statistics, Texture (GLCM/LBP), and Lesion Geometry.")
        
        extracted_feats = extract_handcrafted_features(prep_res['enhanced'], prep_res['lesion_mask'])
        
        f_col1, f_col2 = st.columns([1, 1.2])
        with f_col1:
            st.markdown("##### 📋 Extracted Feature Table (Sample)")
            feats_df = pd.DataFrame(list(extracted_feats.items()), columns=['Feature Name', 'Numerical Value'])
            feats_df['Numerical Value'] = feats_df['Numerical Value'].apply(lambda x: round(float(x), 4))
            st.dataframe(feats_df.head(12), height=240, use_container_width=True)
            
        with f_col2:
            st.markdown("##### 📊 Color & Texture Distribution")
            color_keys = [k for k in extracted_feats if 'color_mean' in k or 'color_std' in k]
            if color_keys:
                cf_df = pd.DataFrame({
                    'Feature': color_keys,
                    'Value': [extracted_feats[k] for k in color_keys]
                })
                if HAS_PLOTLY:
                    fig_f = px.bar(cf_df, x='Feature', y='Value', color='Value', color_continuous_scale='teal', height=240)
                    fig_f.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                    st.plotly_chart(fig_f, use_container_width=True)
                else:
                    st.bar_chart(cf_df.set_index('Feature'))

        st.divider()
        # -------------------------------------------------------------
        # STAGE 4 & 5: PREDICTION & EVALUATION
        # -------------------------------------------------------------
        st.markdown('<span class="step-badge">STAGE 4 & 5: Deep & ML Classification + Performance Evaluation</span>', unsafe_allow_html=True)
        
        with st.spinner("Executing CNN inference and generating Grad-CAM explainability..."):
            diag = predict_deep_image(model, device, pil_img)
            
        pred_c = diag['class']
        conf_val = diag['confidence']
        info = ACNE_KNOWLEDGE_BASE[pred_c]
        sev_col = info['severity_color']
        
        res_c1, res_c2 = st.columns([1, 1.2], gap="medium")
        
        with res_c1:
            st.markdown(f"""
            <div style="background: white; border: 2px solid {sev_col}; border-radius: 14px; padding: 1.3rem; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Diagnostic Output</span>
                    <span style="background: {sev_col}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                        {info['severity']}
                    </span>
                </div>
                <h2 style="margin: 8px 0 2px 0; color: #0f172a; font-size: 1.8rem; font-weight: 800;">
                    {pred_c} <span style="font-size: 1.05rem; color: #64748b; font-weight: 500;">({info['medical_name']})</span>
                </h2>
                <div style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px;">
                        <span>Prediction Confidence</span>
                        <span>{conf_val*100:.1f}%</span>
                    </div>
                    <div style="height: 10px; background: #e2e8f0; border-radius: 5px; overflow: hidden;">
                        <div style="width: {conf_val*100}%; height: 100%; background: {sev_col}; border-radius: 5px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### 🔬 Explainable AI: Grad-CAM Lesion Heatmap")
            g_c1, g_c2 = st.columns(2)
            with g_c1:
                st.image(diag['overlay_img'], caption="Attention Overlay", use_container_width=True)
            with g_c2:
                st.image(diag['heatmap_img'], caption="Activation Map", use_container_width=True)

        with res_c2:
            st.markdown("##### 📊 Full Probability Distribution Across 5 Classes")
            prob_df = pd.DataFrame({
                'Class': list(diag['probabilities'].keys()),
                'Confidence (%)': [v * 100 for v in diag['probabilities'].values()]
            }).sort_values('Confidence (%)', ascending=False)
            
            if HAS_PLOTLY:
                fig_p = px.bar(
                    prob_df, x='Confidence (%)', y='Class', orientation='h',
                    color='Confidence (%)', color_continuous_scale='Blues',
                    text=[f"{p:.1f}%" for p in prob_df['Confidence (%)']],
                    height=200
                )
                fig_p.update_layout(margin=dict(l=0, r=20, t=10, b=10), yaxis_title="", coloraxis_showscale=False)
                fig_p.update_traces(textposition='outside')
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                for _, r in prob_df.iterrows():
                    st.write(f"**{r['Class']}**: {r['Confidence (%)']:.1f}%")
                    st.progress(float(r['Confidence (%)'] / 100.0))
                    
            st.markdown("##### 🧴 Recommended Active Ingredients:")
            for act in info['recommended_actives'][:3]:
                st.markdown(f'<span class="ingredient-tag">{act['name']}</span> - *{act['purpose']}*', unsafe_allow_html=True)
                
            st.markdown(f"""
            <div class="routine-box" style="margin-top: 10px;">
                <strong>Clinical Note:</strong> {info['when_to_see_doctor']}
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: Step-by-Step Feature Extractor & Morphology Lab
# ==============================================================================
with tab_features:
    st.markdown("### 🧬 Feature Extraction & Morphological Analysis Laboratory")
    st.markdown("Explore how computer vision extracts quantitative mathematical features from skin lesions.")
    
    sample_img_path = glob.glob("test/*/*.*")[0] if glob.glob("test/*/*.*") else None
    if sample_img_path:
        f_img = Image.open(sample_img_path).convert('RGB')
        f_np = np.array(f_img)
        f_prep = preprocess_skin_lesion(f_np)
        f_feats = extract_handcrafted_features(f_prep['enhanced'], f_prep['lesion_mask'])
        
        lab_c1, lab_c2 = st.columns([1, 1.2])
        with lab_c1:
            st.image(f_prep['enhanced'], caption=f"Analyzed Sample: {os.path.basename(sample_img_path)}", use_container_width=True)
            st.image(f_prep['lesion_mask'], caption="Extracted Lesion Mask (Otsu + Morphological Cleaning)", use_container_width=True)
            
        with lab_c2:
            st.markdown("#### 📐 Complete Extracted Feature Vector")
            st.write(f"Total features extracted per lesion: **{len(f_feats)} dimensions**")
            
            df_all_f = pd.DataFrame(list(f_feats.items()), columns=['Descriptor Name', 'Extracted Value'])
            st.dataframe(df_all_f, height=380, use_container_width=True)
            
            # Download CSV of features
            st.download_button(
                label="📥 Export Extracted Features (.CSV)",
                data=df_all_f.to_csv(index=False),
                file_name="lesion_extracted_features.csv",
                mime="text/csv"
            )


# ==============================================================================
# TAB 3: Multi-Model ML Training & Evaluation Arena
# ==============================================================================
with tab_ml_arena:
    st.markdown("### 🤖 Multi-Model Machine Learning Benchmark Arena")
    st.markdown("Train, evaluate, and compare **Random Forest**, **Support Vector Machine (SVM)**, **Gradient Boosting**, **KNN**, and **Deep CNN**.")
    
    if os.path.exists("test_metrics.json"):
        with open("test_metrics.json") as f:
            t_metrics = json.load(f)
            
        st.markdown("#### 🏆 Benchmark Performance on Test Dataset (918 Images)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Deep CNN Accuracy", f"{t_metrics['test_accuracy']:.2f}%")
        with mc2:
            st.metric("Weighted F1-Score", f"{t_metrics['weighted_f1']:.2f}%")
        with mc3:
            st.metric("Weighted Precision", f"{t_metrics['weighted_precision']:.2f}%")
        with mc4:
            st.metric("Weighted Recall", f"{t_metrics['weighted_recall']:.2f}%")
            
        st.markdown("#### 🎯 Per-Class Precision & Recall Comparison")
        p_df = pd.DataFrame(t_metrics['per_class']).T
        p_df.columns = ['Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Sample Count']
        st.dataframe(p_df.style.highlight_max(axis=0, color="#d1fae5"), use_container_width=True)
        
        if os.path.exists("confusion_matrix.png"):
            st.markdown("#### 🧩 Multi-Class Confusion Matrix Heatmap")
            st.image("confusion_matrix.png", use_container_width=True)
            
    st.divider()
    st.markdown("#### ⚡ Train Classical ML Classifiers on Handcrafted Features")
    st.caption("Extracts GLCM, Color Moments, and Shape features from dataset and trains Random Forest, SVM, KNN, and Gradient Boosting.")
    
    samples_per_class = st.slider("Samples to extract per class for fast ML training:", min_value=10, max_value=100, value=30, step=10)
    
    if st.button("🚀 Train Classical ML Models Now"):
        with st.status("Extracting features and training ML classifiers...", expanded=True) as status:
            st.write("Extracting multi-modal features from training split...")
            X_tr, y_tr = extract_dataset_features('train', max_samples_per_class=samples_per_class)
            st.write(f"Extracted {X_tr.shape[0]} samples with {X_tr.shape[1]} features!")
            
            st.write("Extracting validation features...")
            X_va, y_val = extract_dataset_features('valid', max_samples_per_class=15)
            
            st.write("Fitting SVM, Random Forest, Gradient Boosting, KNN, and Logistic Regression...")
            ml_results = train_ml_models(X_tr, y_tr, X_va, y_val)
            status.update(label="Classical ML Training Completed!", state="complete", expanded=False)
            
        # Display ML comparison table
        comp_data = []
        for m_name, m_res in ml_results.items():
            comp_data.append({
                'Algorithm': m_name,
                'Train Acc (%)': m_res['train_acc'],
                'Val Acc (%)': m_res['val_acc'],
                'Val F1 (%)': m_res['val_f1'],
                'Val Precision (%)': m_res['val_precision']
            })
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp.style.highlight_max(axis=0, color="#d1fae5"), use_container_width=True)


# ==============================================================================
# TAB 4: Live Python Code Playground & Compiler Console
# ==============================================================================
with tab_code_playground:
    st.markdown("### 💻 Interactive In-Browser Python Code Playground & Compiler")
    st.markdown("Type, edit, and compile Python machine learning and image processing code directly inside this application.")
    
    # Pre-built templates
    TEMPLATE_OPTIONS = {
        "1. Complete End-to-End Image Classification Pipeline": """# Complete End-to-End Skin Lesion Pipeline
import glob
from PIL import Image
from pipeline import load_image_and_metadata, preprocess_skin_lesion, extract_handcrafted_features
from model import AcneClassifier, CLASSES, get_inference_transforms
import torch

# 1. Load an image from test dataset
sample_paths = glob.glob('test/*/*.*')
if sample_paths:
    img_path = sample_paths[0]
    print(f"[*] Step 1: Loaded input image: {img_path}")
    pil_img, np_img, meta = load_image_and_metadata(img_path)
    print(f"    Dimensions: {meta['width']}x{meta['height']} | Brightness: {meta['brightness']}")

    # 2. Preprocessing
    prep = preprocess_skin_lesion(np_img, clip_limit=2.5, blur_ksize=5)
    print(f"[*] Step 2: Preprocessing complete (Denoised, CLAHE, Otsu Mask created)")

    # 3. Feature Extraction
    features = extract_handcrafted_features(prep['enhanced'], prep['lesion_mask'])
    print(f"[*] Step 3: Extracted {len(features)} handcrafted features!")
    print(f"    Sample features: {list(features.items())[:3]}")

    # 4. Deep Learning Classification
    model = AcneClassifier(num_classes=5, pretrained=False)
    checkpoint = torch.load('acne_classifier.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = get_inference_transforms()
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    print(f"[*] Step 4 & 5: Prediction Result = {CLASSES[pred_idx]} ({probs[pred_idx]*100:.1f}% confidence)")
""",
        "2. Custom Feature Extractor (GLCM + Color Moments)": """# Feature Extraction Demonstration
import glob
import numpy as np
from PIL import Image
from pipeline import extract_handcrafted_features, preprocess_skin_lesion

sample_imgs = glob.glob('test/Blackheads/*.*')
if sample_imgs:
    img = Image.open(sample_imgs[0]).convert('RGB')
    np_img = np.array(img)
    
    prep = preprocess_skin_lesion(np_img)
    feats = extract_handcrafted_features(prep['enhanced'], prep['lesion_mask'])
    
    print("=== EXTRACTED FEATURE VECTOR ===")
    for k, v in list(feats.items())[:10]:
        print(f"  {k:28} : {v:.4f}")
    print("================================")
""",
        "3. Train & Evaluate Support Vector Machine (SVM) Classifier": """# Train Classical Machine Learning (SVM & Random Forest)
from pipeline import extract_dataset_features, train_ml_models
import pandas as pd

print("[*] Extracting features from train dataset (20 samples/class for fast demo)...")
X_train, y_train = extract_dataset_features('train', max_samples_per_class=20)
print(f"[*] Training Feature Matrix Shape: {X_train.shape}")

print("[*] Extracting validation features...")
X_val, y_val = extract_dataset_features('valid', max_samples_per_class=10)

print("[*] Training Classical ML Models (SVM, Random Forest, KNN)...")
results = train_ml_models(X_train, y_train, X_val, y_val)

print("\n=== MODEL PERFORMANCE COMPARISON ===")
for name, res in results.items():
    print(f"{name:32} | Train Acc: {res['train_acc']:.1f}% | Val Acc: {res['val_acc']:.1f}% | Val F1: {res['val_f1']:.1f}%")
""",
        "4. Image Preprocessing & CLAHE Filter Comparison (Plot Figure)": """# Visualize Preprocessing Steps with Matplotlib
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pipeline import preprocess_skin_lesion

samples = glob.glob('test/*/*.*')
if samples:
    np_img = np.array(Image.open(samples[0]).convert('RGB'))
    prep = preprocess_skin_lesion(np_img, clip_limit=3.0)
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(prep['original'])
    axes[0].set_title("1. Original RGB")
    axes[0].axis('off')
    
    axes[1].imshow(prep['denoised'])
    axes[1].set_title("2. Denoised")
    axes[1].axis('off')
    
    axes[2].imshow(prep['enhanced'])
    axes[2].set_title("3. CLAHE Enhanced")
    axes[2].axis('off')
    
    axes[3].imshow(prep['segmented_lesion'])
    axes[3].set_title("4. Segmented Lesion")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("[*] Preprocessing visual comparison generated!")
"""
    }
    
    selected_template = st.selectbox("Select Runnable Code Template:", list(TEMPLATE_OPTIONS.keys()))
    
    code_to_run = st.text_area(
        "✏️ Python Code Editor (Modify or type custom program below):",
        value=TEMPLATE_OPTIONS[selected_template],
        height=320
    )
    
    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        run_code_trigger = st.button("⚡ Compile & Run Code", type="primary")
        
    if run_code_trigger:
        st.markdown("##### 🖥️ Execution Console Output (`stdout` / `stderr`):")
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        t0 = time.time()
        
        # Clear existing figures
        plt.close('all')
        
        # Execute code in safe dictionary context
        exec_globals = {
            '__name__': '__main__',
            'plt': plt,
            'np': np,
            'pd': pd,
            'torch': torch,
            'Image': Image,
            'CLASSES': CLASSES,
            'AcneClassifier': AcneClassifier,
            'get_inference_transforms': get_inference_transforms,
            'load_image_and_metadata': load_image_and_metadata,
            'preprocess_skin_lesion': preprocess_skin_lesion,
            'extract_handcrafted_features': extract_handcrafted_features,
            'extract_dataset_features': extract_dataset_features,
            'train_ml_models': train_ml_models
        }
        
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code_to_run, exec_globals)
            exec_time = (time.time() - t0) * 1000
            error_occurred = False
        except Exception as e:
            exec_time = (time.time() - t0) * 1000
            stderr_capture.write(f"\n[Execution Error]: {type(e).__name__}: {str(e)}\n")
            error_occurred = True
            
        output_str = stdout_capture.getvalue()
        error_str = stderr_capture.getvalue()
        
        full_log = (output_str if output_str else "") + (f"\n--- STDERR ---\n{error_str}" if error_str else "")
        if not full_log.strip():
            full_log = "[Code executed successfully with no print output.]"
            
        st.markdown(f'<div class="terminal-box">{full_log}</div>', unsafe_allow_html=True)
        st.caption(f"⏱️ Compilation & Execution Time: **{exec_time:.2f} ms**")
        
        # If any matplotlib figure was created, render it
        if plt.get_fignums():
            st.markdown("##### 📈 Rendered Plot Output:")
            st.pyplot(plt.gcf())


# ==============================================================================
# TAB 5: CLI Compilation & Execution Guide
# ==============================================================================
with tab_docs:
    st.markdown("### 📖 Step-by-Step CLI Compilation & Execution Guide")
    st.markdown("""
    This guide explains how to compile, execute, and evaluate each component of the Skin Lesion Machine Learning pipeline from any terminal or PowerShell window.
    
    ---
    
    #### 1. Launch the All-In-One Web Application
    ```powershell
    python -m streamlit run app.py
    ```
    *Or double-click [`run.bat`](file:///c:/Users/Lenovo/Downloads/archive/AcneDataset/run.bat).*
    
    ---
    
    #### 2. Train / Fine-Tune the Deep Convolutional Neural Network
    ```powershell
    python train.py --epochs 4 --batch_size 32 --lr 0.001 --output acne_classifier.pth
    ```
    - `--epochs`: Number of training epochs (default: `4`)
    - `--batch_size`: Batch size for DataLoader (default: `32`)
    - `--lr`: Initial learning rate (default: `0.001`)
    - `--output`: Checkpoint file path for best weights (default: `acne_classifier.pth`)
    
    ---
    
    #### 3. Evaluate Model on the 918 Test Images
    ```powershell
    python evaluate.py
    ```
    *Generates [`test_metrics.json`](file:///c:/Users/Lenovo/Downloads/archive/AcneDataset/test_metrics.json) and [`confusion_matrix.png`](file:///c:/Users/Lenovo/Downloads/archive/AcneDataset/confusion_matrix.png).*
    
    ---
    
    #### 4. Run Classical ML Feature Extraction & Training Pipeline
    ```powershell
    python -c "from pipeline import extract_dataset_features, train_ml_models; X, y = extract_dataset_features('train', 30); res = train_ml_models(X, y); print('Models trained successfully!')"
    ```
    """)

st.divider()
st.caption("🔬 DermaScan AI • Unified Machine Learning & Computer Vision Platform")
