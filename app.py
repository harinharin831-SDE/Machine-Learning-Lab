"""
=============================================================================
KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING
High-End Medical-AI Diagnostic Dashboard (Streamlit Web Application)
=============================================================================
An end-to-end interactive, educational Machine Learning web application
demonstrating the complete image classification pipeline for renal ultrasound:
  1. Dataset loading, class distribution & balanced analytics
  2. 5-Stage Digital Image Preprocessing (Standardize, Gray, Denoise, Enhance, Normalize)
  3. Hybrid Feature Extraction (HOG Edge Gradients + GLCM Texture Descriptors)
  4. Model Training Studio (SVM, Random Forest, KNN with StandardScaler)
  5. Comprehensive Performance Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
  6. Live Single-Image Testing, 1-Click Sample Testing & Diagnostic AI Report
  7. System Architecture & Model Metadata Specifications

DISCLAIMER: Strictly for academic and research demonstration. Not for clinical diagnosis.
=============================================================================
"""

import os
import io
import time
import base64
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import modular project components
from preprocessing import preprocess_image, calculate_image_histogram, TARGET_IMAGE_SIZE
from feature_extraction import (
    extract_features,
    extract_hog_features,
    extract_glcm_features,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    GLCM_PROPERTIES
)
from model_training import (
    train_model,
    save_model,
    load_saved_model,
    SUPPORTED_ALGORITHMS,
    DEFAULT_MODEL_PATH
)
from evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
    save_evaluation_outputs
)
from prediction import predict_single_image, DISCLAIMER_TEXT
from utils import (
    get_dataset_summary,
    load_dataset,
    split_data,
    create_sample_synthetic_dataset
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="RenalAI - Kidney Disease ML Diagnostic Lab",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM ULTRA-PREMIUM MEDICAL-TECH CSS & ICONOGRAPHY
# =============================================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    :root {
        --primary-blue: #0284C7;
        --primary-hover: #0369A1;
        --accent-cyan: #38BDF8;
        --accent-indigo: #6366F1;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
        --bg-dark-card: #0F172A;
        --border-subtle: #E2E8F0;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
        letter-spacing: -0.02em;
    }

    code, pre, .stCodeBlock, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1360px;
    }

    /* Premium Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0A1128 0%, #101F42 45%, #0B3C68 100%);
        color: white;
        padding: 2.2rem 2.2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 35px -8px rgba(10, 17, 40, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, rgba(56, 189, 248, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-title-wrap {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 0.6rem;
    }

    .hero-icon-badge {
        width: 54px;
        height: 54px;
        background: linear-gradient(135deg, #38BDF8 0%, #2563EB 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        box-shadow: 0 8px 16px rgba(37, 99, 235, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        color: #FFFFFF;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #94A3B8;
        margin: 0 0 1.2rem 0;
        max-width: 850px;
    }

    .hero-badges-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
        color: #E2E8F0;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .hero-pill.highlight {
        background: rgba(56, 189, 248, 0.18);
        color: #38BDF8;
        border-color: rgba(56, 189, 248, 0.35);
    }

    .hero-pill.live {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.35);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        display: inline-block;
        animation: pulseDot 2s infinite;
    }

    @keyframes pulseDot {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Clinical Disclaimer Banner */
    .clinical-disclaimer-card {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left: 5px solid #F59E0B;
        border: 1px solid #FDE68A;
        border-left-width: 5px;
        color: #92400E;
        padding: 1rem 1.35rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.08);
        display: flex;
        gap: 14px;
        align-items: flex-start;
    }

    .disclaimer-icon-wrap {
        font-size: 1.5rem;
        line-height: 1;
        margin-top: 2px;
    }

    .disclaimer-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #B45309;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .disclaimer-body {
        font-size: 0.88rem;
        line-height: 1.5;
        color: #78350F;
    }

    /* Interactive Pipeline Progress Ribbon */
    .pipeline-ribbon {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        overflow-x: auto;
    }

    .ribbon-step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.83rem;
        font-weight: 600;
        color: #475569;
        padding: 0.4rem 0.75rem;
        border-radius: 8px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        white-space: nowrap;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .ribbon-step.active {
        background: #EFF6FF;
        color: #1D4ED8;
        border-color: #BFDBFE;
    }

    .ribbon-step-num {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #E2E8F0;
        color: #475569;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .ribbon-step.active .ribbon-step-num {
        background: #2563EB;
        color: #FFFFFF;
    }

    .ribbon-arrow {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 800;
    }

    /* Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.35rem 1.15rem;
        text-align: center;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.03);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #38BDF8, #2563EB);
    }

    .kpi-card.acc::before { background: linear-gradient(90deg, #38BDF8, #2563EB); }
    .kpi-card.prec::before { background: linear-gradient(90deg, #34D399, #059669); }
    .kpi-card.rec::before { background: linear-gradient(90deg, #A78BFA, #7C3AED); }
    .kpi-card.f1::before { background: linear-gradient(90deg, #F472B6, #DB2777); }

    .kpi-icon {
        font-size: 1.4rem;
        margin-bottom: 0.35rem;
    }

    .kpi-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }

    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        font-family: 'Outfit', sans-serif;
        line-height: 1.1;
    }

    .kpi-value.acc { color: #1D4ED8; }
    .kpi-value.prec { color: #059669; }
    .kpi-value.rec { color: #6D28D9; }
    .kpi-value.f1 { color: #BE185D; }

    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* Modern Feature Cards */
    .tech-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.35rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .tech-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
    }

    .tech-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.85rem;
    }

    .tech-card-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }

    .tech-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }

    /* Stage Progression Cards */
    .stage-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.85rem 0.65rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
    }

    .stage-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }

    .stage-badge-top {
        display: inline-block;
        background: #F1F5F9;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stage-title {
        font-size: 0.86rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.35rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Diagnosis Result Card */
    .diagnosis-result-card {
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        border-width: 2px;
        border-style: solid;
        position: relative;
    }

    .diag-Normal {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-color: #10B981;
        color: #065F46;
    }

    .diag-Cyst {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-color: #3B82F6;
        color: #1E40AF;
    }

    .diag-Stone {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-color: #F59E0B;
        color: #92400E;
    }

    .diag-Tumor {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-color: #EF4444;
        color: #991B1B;
    }

    .diag-Other {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        border-color: #64748B;
        color: #1E293B;
    }

    .diag-header-label {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }

    .diag-class-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0.2rem 0;
        font-family: 'Outfit', sans-serif;
    }

    .diag-confidence-meter {
        margin-top: 0.75rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.7);
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* Class Badge Pills */
    .class-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .class-pill.Normal { background: #DCFCE7; color: #15803D; }
    .class-pill.Cyst { background: #DBEAFE; color: #1D4ED8; }
    .class-pill.Stone { background: #FEF3C7; color: #B45309; }
    .class-pill.Tumor { background: #FEE2E2; color: #B91C1C; }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #475569;
        background-color: transparent;
        border: none;
        padding: 0 14px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0284C7 !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    }

    /* App Footer */
    .app-footer {
        margin-top: 3.5rem;
        padding-top: 1.75rem;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* Custom Quick Sample Button Cards */
    .sample-btn-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "dataset_data" not in st.session_state:
    st.session_state["dataset_data"] = None

if "split_data" not in st.session_state:
    st.session_state["split_data"] = None

if "model_package" not in st.session_state:
    # Automatically load existing trained model if present
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            st.session_state["model_package"] = load_saved_model(DEFAULT_MODEL_PATH)
        except Exception:
            st.session_state["model_package"] = None
    else:
        st.session_state["model_package"] = None

if "eval_results" not in st.session_state:
    st.session_state["eval_results"] = None

if "active_tab_idx" not in st.session_state:
    st.session_state["active_tab_idx"] = 0

if "selected_preset_img" not in st.session_state:
    st.session_state["selected_preset_img"] = None


# =============================================================================
# HERO BANNER & STATUS BAR
# =============================================================================
model_is_ready = st.session_state["model_package"] is not None
data_is_ready = st.session_state["dataset_data"] is not None
active_algo_name = st.session_state["model_package"].get("algorithm", "None") if model_is_ready else "Not Loaded"
sample_count_str = f"{st.session_state['dataset_data']['total_images']} Images" if data_is_ready else "0 Images"

st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
            <div class="hero-title-wrap">
                <div class="hero-icon-badge">🫘</div>
                <div>
                    <h1 class="hero-title">KIDNEY DISEASE ML DIAGNOSTIC PLATFORM</h1>
                </div>
            </div>
            <div class="hero-subtitle">
                Academic & Educational Machine Learning Studio for Renal Ultrasound Image Analysis,
                5-Stage Feature Engineering, Multiclass Classification, and Confidence Estimation.
            </div>
            <div class="hero-badges-row">
                <span class="hero-pill live">
                    <span class="status-dot"></span> System Live & Operational
                </span>
                <span class="hero-pill highlight">
                    🧬 Hybrid HOG (8,100D) + GLCM (60D) = 8,160D
                </span>
                <span class="hero-pill">
                    🤖 Active Model: <strong>{active_algo_name}</strong>
                </span>
                <span class="hero-pill">
                    📁 Dataset: <strong>{sample_count_str}</strong>
                </span>
                <span class="hero-pill">
                    ⚡ Spatial Standard: <strong>128×128 px</strong>
                </span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# CLINICAL & EDUCATIONAL SAFETY NOTICE
# =============================================================================
st.markdown("""
<div class="clinical-disclaimer-card">
    <div class="disclaimer-icon-wrap">🛡️</div>
    <div>
        <div class="disclaimer-title">
            <strong>Mandatory Academic & Educational Safety Notice</strong>
        </div>
        <div class="disclaimer-body">
            This platform is an <strong>educational and research demonstration</strong> developed to showcase traditional
            Computer Vision and Machine Learning algorithms (HOG, GLCM, SVM, Random Forest, KNN).
            It is <strong>NOT</strong> a certified medical diagnostic device and must <strong>NEVER</strong> be utilized for
            clinical decision-making, patient triage, or disease diagnosis. Always consult certified radiologists and healthcare practitioners.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# INTERACTIVE WORKFLOW PIPELINE BREADCRUMB
# =============================================================================
st.markdown("""
<div class="pipeline-ribbon">
    <div class="ribbon-step active">
        <div class="ribbon-step-num">1</div>
        <span>📂 1. Dataset Explorer</span>
    </div>
    <div class="ribbon-arrow">➔</div>
    <div class="ribbon-step">
        <div class="ribbon-step-num">2</div>
        <span>🔬 2. Preprocessing Lab</span>
    </div>
    <div class="ribbon-arrow">➔</div>
    <div class="ribbon-step">
        <div class="ribbon-step-num">3</div>
        <span>🧬 3. Feature Extraction</span>
    </div>
    <div class="ribbon-arrow">➔</div>
    <div class="ribbon-step">
        <div class="ribbon-step-num">4</div>
        <span>🤖 4. Model Training</span>
    </div>
    <div class="ribbon-arrow">➔</div>
    <div class="ribbon-step">
        <div class="ribbon-step-num">5</div>
        <span>📊 5. Evaluation</span>
    </div>
    <div class="ribbon-arrow">➔</div>
    <div class="ribbon-step active">
        <div class="ribbon-step-num">6</div>
        <span>🎯 6. Diagnostic Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR CONTROLS & ENVIRONMENT STATUS
# =============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Environment Status Card
    st.markdown(f"""
    <div class="tech-card" style="padding: 1rem; margin-bottom: 1rem; border-color: #CBD5E1;">
        <div style="font-size:0.8rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:0.5rem; display:flex; align-items:center; gap:6px;">
            <span>⚡ System Live Status</span>
        </div>
        <div style="font-size:0.85rem; line-height: 1.8;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Dataset:</span>
                <strong>{'🟢 Loaded' if data_is_ready else '⚪ Not Loaded'}</strong>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Trained Model:</span>
                <strong>{'🟢 Active' if model_is_ready else '⚪ Not Active'}</strong>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Algorithm:</span>
                <span class="mono-font" style="font-size:0.8rem; font-weight:600; color:#0284C7;">{active_algo_name}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748B;">Features:</span>
                <span class="mono-font" style="font-size:0.8rem;">8,160D Vector</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📂 Dataset Configuration")
    dataset_path_input = st.text_input(
        "Dataset Directory Path",
        value="dataset",
        help="Path to folder containing class subdirectories: Normal, Cyst, Stone, Tumor."
    )

    # 1-Click Synthetic Demo Generator
    if st.button("🧪 Generate Synthetic Demo Data", use_container_width=True, help="Create realistic synthetic ultrasound images for testing."):
        with st.spinner("Generating synthetic ultrasound dataset..."):
            created = create_sample_synthetic_dataset(output_dir=dataset_path_input, images_per_class=15)
            st.success(f"Generated {sum(created.values())} synthetic samples in `{dataset_path_input}`!")
            time.sleep(1)
            st.rerun()

    # 1-Click Quick Start Auto-Pipeline
    if st.button("⚡ 1-Click Auto Demo (Load + Train SVM)", type="secondary", use_container_width=True):
        with st.spinner("Executing Automated End-to-End Pipeline (Load -> Extract -> Train -> Evaluate)..."):
            # Ensure dataset exists
            if not os.path.exists(dataset_path_input):
                create_sample_synthetic_dataset(output_dir=dataset_path_input, images_per_class=15)
            
            # Load dataset
            data = load_dataset(dataset_path_input)
            st.session_state["dataset_data"] = data
            
            # Split
            X_tr, X_te, y_tr, y_te, is_strat = split_data(data["X"], data["y"], test_size=0.20, random_state=42)
            st.session_state["split_data"] = {
                "X_train": X_tr, "X_test": X_te,
                "y_train": y_tr, "y_test": y_te,
                "is_stratified": is_strat, "test_size": 0.20
            }
            
            # Train default SVM
            model_pkg = train_model(
                X_train=X_tr, y_train=y_tr,
                algorithm="SVM", params={"C": 1.0, "kernel": "rbf", "gamma": "scale"},
                class_names=data["class_names"]
            )
            
            # Evaluate
            eval_res = evaluate_model(model_pkg, X_te, y_te, data["class_names"])
            model_pkg["test_metrics"] = {
                "accuracy": eval_res["accuracy"] * 100,
                "precision": eval_res["precision"] * 100,
                "recall": eval_res["recall"] * 100,
                "f1_score": eval_res["f1_score"] * 100
            }
            save_model(model_pkg, DEFAULT_MODEL_PATH)
            save_evaluation_outputs(eval_res, "outputs")
            
            st.session_state["model_package"] = model_pkg
            st.session_state["eval_results"] = eval_res
            st.success(f"🎉 Pipeline Ready! Accuracy: {eval_res['accuracy']*100:.1f}%")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🔬 Pipeline Descriptors")
    st.markdown("""
    - **HOG Gradients:** 9 Bins, 8×8 Cells, 2×2 Blocks *(8,100D)*
    - **GLCM Texture:** 5 Properties, 3 Distances, 4 Angles *(60D)*
    - **Total Combined:** **8,160 Descriptors**
    """)

    st.markdown("---")
    st.markdown("#### 📖 Project Information")
    st.caption("Developed with Python 3.10+, Scikit-Learn, OpenCV, Scikit-Image, and Streamlit.")


# =============================================================================
# MAIN NAVIGATION TABS
# =============================================================================
tabs = st.tabs([
    "🏠 Overview & Architecture",
    "📊 Dataset Explorer",
    "🔬 5-Stage Preprocessor",
    "🧬 Feature Extraction Studio",
    "🤖 Model Training Studio",
    "📈 Performance Evaluation",
    "🎯 Live Diagnostic Inference",
    "ℹ️ Model Specs & Info"
])


# =============================================================================
# TAB 1: 🏠 OVERVIEW & PIPELINE ARCHITECTURE
# =============================================================================
with tabs[0]:
    st.markdown("### 📌 End-to-End Pipeline & System Architecture")
    
    col1, col2 = st.columns([1.6, 1.4])
    with col1:
        st.markdown("""
        Ultrasound imaging is the standard non-invasive diagnostic modality for evaluating renal parenchymal abnormalities.
        This academic project demonstrates how **Digital Image Processing** and **Traditional Machine Learning** can be
        transparently orchestrated to classify ultrasound images into anatomical categories:
        `Normal`, `Cyst`, `Stone`, and `Tumor`.
        
        #### 🎯 Core Engineering Principles:
        1. **Transparent White-Box Features:** Eliminates opaque deep learning representations in favor of mathematically defined morphological gradients (**HOG**) and statistical texture co-occurrences (**GLCM**).
        2. **Standardized Preprocessing:** Mitigates ultrasound speckle noise and acoustic attenuation through Gaussian filtering and histogram equalization.
        3. **Deterministic Classification:** Robust multi-class modeling using Support Vector Machines (SVM), Random Forest ensembles, and K-Nearest Neighbors (KNN).
        4. **Production Readiness:** Full model serialization via `joblib` and automated diagnostic report generation.
        """)

    with col2:
        st.markdown("""
        <div class="tech-card" style="background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%); border-color: #BFDBFE;">
            <div class="tech-card-header">
                <div class="tech-card-icon" style="background:#DBEAFE; color:#1E40AF;">🏗️</div>
                <h4 class="tech-card-title" style="color:#1E40AF;">System Architecture Flow</h4>
            </div>
            <div style="font-size:0.88rem; line-height: 1.8; color:#334155;">
                <strong>1. Input Acquisition:</strong> 128×128 Ultrasound Scan<br>
                <strong>2. Preprocessing:</strong> Gaussian Blur + HistEq + MinMax<br>
                <strong>3. Feature Extraction:</strong> HOG (8,100D) + GLCM (60D) = 8,160D<br>
                <strong>4. Standardization:</strong> Scikit-Learn <code>StandardScaler</code><br>
                <strong>5. ML Inference:</strong> SVM / Random Forest / KNN<br>
                <strong>6. Output Verdict:</strong> Class Label + Calibrated Confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔬 Supported Machine Learning Classifiers")
    
    algo_col1, algo_col2, algo_col3 = st.columns(3)
    with algo_col1:
        st.markdown("""
        <div class="tech-card" style="height: 100%;">
            <div class="tech-card-header">
                <div class="tech-card-icon" style="background:#EFF6FF; color:#2563EB;">⚡</div>
                <h4 class="tech-card-title" style="color:#1E40AF;">1. Support Vector Machine</h4>
            </div>
            <p style="font-size:0.86rem; color:#475569; line-height:1.5;">
                Constructs maximum-margin hyperplanes in a high-dimensional kernel space (RBF). Highly effective for 8,160-dimensional feature spaces.
            </p>
            <div style="font-size:0.82rem; color:#64748B; background:#F8FAFC; padding:0.6rem; border-radius:8px; border:1px solid #E2E8F0;">
                <div><strong>Kernel:</strong> RBF (Radial Basis Function)</div>
                <div><strong>Probability:</strong> Platt Scaling Calibration</div>
                <div><strong>Advantage:</strong> Resists overfitting in high dimensions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with algo_col2:
        st.markdown("""
        <div class="tech-card" style="height: 100%;">
            <div class="tech-card-header">
                <div class="tech-card-icon" style="background:#ECFDF5; color:#059669;">🌲</div>
                <h4 class="tech-card-title" style="color:#065F46;">2. Random Forest Ensemble</h4>
            </div>
            <p style="font-size:0.86rem; color:#475569; line-height:1.5;">
                An ensemble of 100 decorrelated decision trees built via bootstrap aggregating (bagging) and random feature subspace partitioning.
            </p>
            <div style="font-size:0.82rem; color:#64748B; background:#F8FAFC; padding:0.6rem; border-radius:8px; border:1px solid #E2E8F0;">
                <div><strong>Estimators:</strong> 100 Decision Trees</div>
                <div><strong>Probability:</strong> Mean class vote across forest</div>
                <div><strong>Advantage:</strong> Captures complex non-linear boundaries</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with algo_col3:
        st.markdown("""
        <div class="tech-card" style="height: 100%;">
            <div class="tech-card-header">
                <div class="tech-card-icon" style="background:#F5F3FF; color:#7C3AED;">🎯</div>
                <h4 class="tech-card-title" style="color:#5B21B6;">3. K-Nearest Neighbors</h4>
            </div>
            <p style="font-size:0.86rem; color:#475569; line-height:1.5;">
                Non-parametric instance-based classifier assigning class labels based on standardized Euclidean distance to the K nearest training instances.
            </p>
            <div style="font-size:0.82rem; color:#64748B; background:#F8FAFC; padding:0.6rem; border-radius:8px; border:1px solid #E2E8F0;">
                <div><strong>Neighbors:</strong> K = 5</div>
                <div><strong>Metric:</strong> Standardized Euclidean Distance</div>
                <div><strong>Advantage:</strong> Intuitive geometric decision boundaries</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: 📊 DATASET EXPLORER
# =============================================================================
with tabs[1]:
    st.markdown("### 📊 Dataset Exploration & Class Distribution Analytics")

    summary = get_dataset_summary(dataset_path_input)

    if not summary["exists"]:
        st.warning(f"⚠️ Dataset path `{dataset_path_input}` not found. Click **'🧪 Generate Synthetic Demo Data'** in the sidebar to create sample images.")
    else:
        # Top KPI Metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="kpi-card acc">
                <div class="kpi-icon">📁</div>
                <div class="kpi-label">Total Images</div>
                <div class="kpi-value">{summary['total_images']}</div>
                <div class="kpi-sub">Ultrasound Scans</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="kpi-card prec">
                <div class="kpi-icon">🏷️</div>
                <div class="kpi-label">Target Classes</div>
                <div class="kpi-value prec">{summary['num_classes']}</div>
                <div class="kpi-sub">{', '.join(summary['class_names'])}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            status_color = "#10B981" if st.session_state["dataset_data"] else "#64748B"
            st.markdown(f"""
            <div class="kpi-card rec">
                <div class="kpi-icon">⚡</div>
                <div class="kpi-label">Feature Status</div>
                <div class="kpi-value rec" style="font-size:1.6rem;">
                    {'Ready' if st.session_state['dataset_data'] else 'Pending'}
                </div>
                <div class="kpi-sub">8,160D Vectors</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="kpi-card f1">
                <div class="kpi-icon">📐</div>
                <div class="kpi-label">Resolution</div>
                <div class="kpi-value f1" style="font-size:1.6rem;">128×128</div>
                <div class="kpi-sub">Spatial Standard</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Action Buttons
        btn_c1, btn_c2 = st.columns([1.5, 2.5])
        with btn_c1:
            if st.button("🚀 Load & Extract Features for All Images", type="primary", use_container_width=True):
                p_bar = st.progress(0.0)
                s_txt = st.empty()

                def progress_fn(ratio, text):
                    p_bar.progress(ratio)
                    s_txt.text(text)

                try:
                    with st.spinner("Extracting HOG (8,100D) + GLCM (60D) descriptors across all images..."):
                        data = load_dataset(dataset_path_input, progress_callback=progress_fn)
                        st.session_state["dataset_data"] = data
                        
                        X_tr, X_te, y_tr, y_te, is_strat = split_data(data["X"], data["y"], test_size=0.20, random_state=42)
                        st.session_state["split_data"] = {
                            "X_train": X_tr, "X_test": X_te,
                            "y_train": y_tr, "y_test": y_te,
                            "is_stratified": is_strat, "test_size": 0.20
                        }
                    s_txt.empty()
                    p_bar.empty()
                    st.success(f"✅ Successfully extracted features from {data['total_images']} images across classes: {', '.join(data['class_names'])}!")
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")

        # Class Distribution Analytics Charts
        if summary["num_classes"] > 0:
            st.markdown("---")
            st.markdown("#### 📈 Class Distribution & Balance")
            chart_col1, chart_col2 = st.columns(2)
            
            counts_df = pd.DataFrame(
                list(summary["class_counts"].items()),
                columns=["Class", "Image Count"]
            )

            palette_map = {
                "Normal": "#10B981",
                "Cyst": "#3B82F6",
                "Stone": "#F59E0B",
                "Tumor": "#EF4444"
            }
            bar_colors = [palette_map.get(c, "#6366F1") for c in counts_df["Class"]]

            with chart_col1:
                fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
                sns.barplot(data=counts_df, x="Class", y="Image Count", palette=bar_colors, ax=ax)
                ax.set_title("Sample Counts per Class", fontweight="bold", fontsize=11, pad=12)
                ax.set_ylabel("Number of Scans", fontsize=9, fontweight="bold")
                ax.set_xlabel("Diagnostic Category", fontsize=9, fontweight="bold")
                for p in ax.patches:
                    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=9, fontweight="bold", xytext=(0, 4), textcoords='offset points')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with chart_col2:
                fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
                ax.pie(
                    counts_df["Image Count"],
                    labels=counts_df["Class"],
                    autopct="%1.1f%%",
                    colors=bar_colors,
                    startangle=140,
                    wedgeprops={"edgecolor": "white", "linewidth": 2.5}
                )
                ax.set_title("Class Proportion Distribution", fontweight="bold", fontsize=11, pad=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Dataset Sample Previews
        st.markdown("---")
        st.markdown("#### 🖼️ Visual Class Gallery (Representative Ultrasound Samples)")
        if summary["class_names"]:
            preview_cols = st.columns(len(summary["class_names"]))
            for idx, c_name in enumerate(summary["class_names"]):
                with preview_cols[idx]:
                    badge_cls = c_name if c_name in ["Normal", "Cyst", "Stone", "Tumor"] else "Other"
                    st.markdown(f"<div class='class-pill {badge_cls}'>{c_name}</div>", unsafe_allow_html=True)
                    c_folder = os.path.join(dataset_path_input, c_name)
                    c_files = [f for f in os.listdir(c_folder) if not f.startswith(".")]
                    if c_files:
                        img_path = os.path.join(c_folder, c_files[0])
                        st.image(img_path, caption=f"{c_name} Scan (Sample 1)", use_container_width=True)


# =============================================================================
# TAB 3: 🔬 5-STAGE PREPROCESSOR LAB
# =============================================================================
with tabs[2]:
    st.markdown("### 🔬 5-Stage Digital Image Preprocessing Lab")
    st.markdown("""
    Ultrasound imaging suffers from inherent acoustic speckle noise, varying operator pressure, and low contrast.
    The 5-stage preprocessing pipeline standardizes spatial dimensions, suppresses noise, and maximizes anatomical edge gradients.
    """)

    prep_c1, prep_c2 = st.columns([1.5, 1.5])
    with prep_c1:
        uploaded_prep_img = st.file_uploader(
            "📤 Upload Image for Preprocessing Inspection:",
            type=["jpg", "jpeg", "png", "bmp", "tif"],
            key="prep_uploader_v2"
        )
    with prep_c2:
        sample_img_choice = None
        if summary["exists"] and summary["total_images"] > 0:
            sample_options = []
            for c_name in summary["class_names"]:
                c_dir = os.path.join(dataset_path_input, c_name)
                for f in os.listdir(c_dir)[:2]:
                    sample_options.append(os.path.join(c_dir, f))
            if sample_options:
                sample_img_choice = st.selectbox(
                    "📁 Or Select Sample from Dataset:",
                    options=["-- Select from Dataset --"] + sample_options,
                    index=1 if uploaded_prep_img is None and len(sample_options) > 0 else 0
                )

    active_prep_input = None
    if uploaded_prep_img is not None:
        active_prep_input = uploaded_prep_img
    elif sample_img_choice and sample_img_choice != "-- Select from Dataset --":
        active_prep_input = sample_img_choice
    elif os.path.exists("sample_images/test_normal_ultrasound.png"):
        active_prep_input = "sample_images/test_normal_ultrasound.png"

    if active_prep_input is not None:
        try:
            prep_res = preprocess_image(active_prep_input)

            st.markdown("---")
            st.markdown("#### 🔍 Step-by-Step Visual Transformation")
            st_col1, st_col2, st_col3, st_col4, st_col5 = st.columns(5)

            with st_col1:
                st.markdown("""
                <div class="stage-card">
                    <span class="stage-badge-top">Stage 01</span>
                    <div class="stage-title">Raw Input</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(prep_res["original"], use_container_width=True)
                st.caption(f"{prep_res['original'].shape[1]}×{prep_res['original'].shape[0]} RGB")

            with st_col2:
                st.markdown("""
                <div class="stage-card">
                    <span class="stage-badge-top">Stage 02</span>
                    <div class="stage-title">Resized & Gray</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(prep_res["gray"], use_container_width=True, clamp=True)
                st.caption("128×128 Spatial Standard")

            with st_col3:
                st.markdown("""
                <div class="stage-card">
                    <span class="stage-badge-top">Stage 03</span>
                    <div class="stage-title">Gaussian Denoised</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(prep_res["denoised"], use_container_width=True, clamp=True)
                st.caption("5×5 Kernel, σ=1.0")

            with st_col4:
                st.markdown("""
                <div class="stage-card">
                    <span class="stage-badge-top">Stage 04</span>
                    <div class="stage-title">Hist. Equalized</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(prep_res["enhanced"], use_container_width=True, clamp=True)
                st.caption("CDF Contrast Boost")

            with st_col5:
                st.markdown("""
                <div class="stage-card">
                    <span class="stage-badge-top">Stage 05</span>
                    <div class="stage-title">Normalized</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(prep_res["normalized_uint8"], use_container_width=True, clamp=True)
                st.caption("Float32 [0.0, 1.0]")

            # Intensity Histogram Transformation
            st.markdown("---")
            st.markdown("#### 📊 Pixel Intensity Histogram Transformation")
            hist_col1, hist_col2 = st.columns(2)

            with hist_col1:
                fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
                h_gray, _ = calculate_image_histogram(prep_res["gray"])
                ax.plot(h_gray, color="#64748B", linewidth=2, label="Original Density")
                ax.fill_between(range(256), h_gray, color="#CBD5E1", alpha=0.5)
                ax.set_title("Before Equalization (Original Grayscale Distribution)", fontsize=10, fontweight="bold")
                ax.set_xlim(0, 255)
                ax.set_xlabel("Pixel Intensity [0 - 255]", fontsize=9)
                ax.set_ylabel("Pixel Frequency", fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with hist_col2:
                fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
                h_enh, _ = calculate_image_histogram(prep_res["enhanced"])
                ax.plot(h_enh, color="#0284C7", linewidth=2, label="Equalized Density")
                ax.fill_between(range(256), h_enh, color="#BAE6FD", alpha=0.5)
                ax.set_title("After Histogram Equalization (Uniform Dynamic Spread)", fontsize=10, fontweight="bold")
                ax.set_xlim(0, 255)
                ax.set_xlabel("Pixel Intensity [0 - 255]", fontsize=9)
                ax.set_ylabel("Pixel Frequency", fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
    else:
        st.info("💡 Upload an image or select a sample above to inspect the preprocessing pipeline.")


# =============================================================================
# TAB 4: 🧬 FEATURE EXTRACTION STUDIO
# =============================================================================
with tabs[3]:
    st.markdown("### 🧬 Feature Extraction: Hybrid HOG + GLCM Descriptors")
    st.markdown("""
    Feature extraction maps spatial image matrices into high-dimensional numerical vectors combining
    **Histogram of Oriented Gradients (HOG)** edge directionality with **Gray-Level Co-occurrence Matrix (GLCM)** second-order texture statistics.
    """)

    feat_src = active_prep_input if active_prep_input is not None else None
    if feat_src is None and summary["exists"] and summary["total_images"] > 0:
        for c in summary["class_names"]:
            c_dir = os.path.join(dataset_path_input, c)
            files = [f for f in os.listdir(c_dir) if not f.startswith(".")]
            if files:
                feat_src = os.path.join(c_dir, files[0])
                break

    if feat_src is not None:
        try:
            prep_img = preprocess_image(feat_src)["enhanced"]
            feat_info = extract_features(prep_img, return_breakdown=True)

            # Feature Dimension Badges
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                st.markdown(f"""
                <div class="kpi-card acc">
                    <div class="kpi-icon">🧭</div>
                    <div class="kpi-label">HOG Features</div>
                    <div class="kpi-value acc">{feat_info['hog_count']:,}</div>
                    <div class="kpi-sub">9 Bins, 8×8 Cells</div>
                </div>
                """, unsafe_allow_html=True)
            with f_col2:
                st.markdown(f"""
                <div class="kpi-card prec">
                    <div class="kpi-icon">🧬</div>
                    <div class="kpi-label">GLCM Features</div>
                    <div class="kpi-value prec">{feat_info['glcm_count']:,}</div>
                    <div class="kpi-sub">5 Props × 3 Dist × 4 Ang</div>
                </div>
                """, unsafe_allow_html=True)
            with f_col3:
                st.markdown(f"""
                <div class="kpi-card rec">
                    <div class="kpi-icon">🔗</div>
                    <div class="kpi-label">Combined Total</div>
                    <div class="kpi-value rec">{feat_info['total_count']:,}</div>
                    <div class="kpi-sub">Fused Feature Vector</div>
                </div>
                """, unsafe_allow_html=True)
            with f_col4:
                st.markdown(f"""
                <div class="kpi-card f1">
                    <div class="kpi-icon">📊</div>
                    <div class="kpi-label">Vector Shape</div>
                    <div class="kpi-value f1" style="font-size:1.5rem;">{feat_info['shape']}</div>
                    <div class="kpi-sub">1D NumPy Array</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Visualizations: HOG Vector Field & GLCM Texture Chart
            h_col, g_col = st.columns([1.2, 1.8])
            with h_col:
                st.markdown("#### 🧭 HOG Gradient Vector Field")
                if feat_info["hog_image"] is not None:
                    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=100)
                    im = ax.imshow(feat_info["hog_image"], cmap="inferno")
                    ax.axis("off")
                    ax.set_title("HOG Gradient Orientations", fontsize=10, fontweight="bold")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            with g_col:
                st.markdown("#### 🧬 GLCM Texture Descriptors Breakdown")
                glcm_df = pd.DataFrame(
                    list(feat_info["glcm_summary"].items()),
                    columns=["Texture Property", "Mean Value"]
                )
                
                fig, ax = plt.subplots(figsize=(6, 3.2), dpi=100)
                sns.barplot(data=glcm_df, x="Texture Property", y="Mean Value", palette="viridis", ax=ax)
                ax.set_title("GLCM Statistical Property Intensities", fontsize=10, fontweight="bold")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(glcm_df.style.format({"Mean Value": "{:.5f}"}), use_container_width=True)

            # Feature Vector Preview
            st.markdown("---")
            st.markdown("#### 🔢 Feature Vector Preview (First 20 Elements)")
            first_20 = feat_info["first_20"]
            f20_df = pd.DataFrame({
                f"f_{i:02d}": [f"{val:.4f}"] for i, val in enumerate(first_20)
            })
            st.dataframe(f20_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error in feature extraction: {e}")
    else:
        st.info("💡 Load a dataset or upload an image to inspect HOG and GLCM feature extraction.")


# =============================================================================
# TAB 5: 🤖 MODEL TRAINING STUDIO
# =============================================================================
with tabs[4]:
    st.markdown("### 🤖 Machine Learning Model Training Studio")
    st.markdown("""
    Train and benchmark classical Machine Learning classifiers with automated `StandardScaler` feature normalization.
    """)

    if st.session_state["dataset_data"] is None:
        st.warning("⚠️ Please load a dataset in the **'Dataset Explorer'** tab or use the sidebar **'1-Click Auto Demo'**.")
    else:
        dataset_data = st.session_state["dataset_data"]
        
        train_cfg_col1, train_cfg_col2 = st.columns([1.2, 1.8])

        with train_cfg_col1:
            st.markdown("#### ⚙️ Algorithm Selection")
            selected_algo = st.selectbox(
                "Select Classifier Algorithm:",
                options=SUPPORTED_ALGORITHMS,
                index=0,
                help="Choose from Support Vector Machine, Random Forest, or K-Nearest Neighbors"
            )

            test_split_ratio = st.slider(
                "Validation Test Split Ratio:",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                help="Recommended: 0.20 (80% Training, 20% Testing)"
            )

        with train_cfg_col2:
            st.markdown("#### 🎛️ Hyperparameter Fine-Tuning")
            algo_params = {}
            if selected_algo == "SVM":
                c_val = st.number_input("C (Regularization Penalty):", min_value=0.01, max_value=100.0, value=1.0, step=0.5)
                kernel_choice = st.selectbox("Kernel Function:", options=["rbf", "linear", "poly", "sigmoid"], index=0)
                algo_params = {"C": c_val, "kernel": kernel_choice, "gamma": "scale"}
            elif selected_algo == "Random Forest":
                n_trees = st.slider("Number of Trees (n_estimators):", min_value=10, max_value=300, value=100, step=10)
                max_d = st.slider("Max Depth (Tree Complexity):", min_value=2, max_value=30, value=15)
                algo_params = {"n_estimators": n_trees, "max_depth": max_d}
            elif selected_algo == "KNN":
                k_val = st.slider("Number of Neighbors (K):", min_value=1, max_value=25, value=5, step=2)
                weights_choice = st.selectbox("Weighting Function:", options=["uniform", "distance"], index=0)
                algo_params = {"n_neighbors": k_val, "weights": weights_choice}

        st.markdown("---")

        if st.button("🚀 Train Model Pipeline Now", type="primary", use_container_width=True):
            train_progress = st.progress(0)
            train_status = st.empty()

            try:
                train_status.info("1/4: Partitioning dataset with stratified sampling...")
                train_progress.progress(25)
                time.sleep(0.2)

                X = dataset_data["X"]
                y = dataset_data["y"]
                class_names = dataset_data["class_names"]

                X_train, X_test, y_train, y_test, is_strat = split_data(
                    X, y, test_size=test_split_ratio, random_state=42
                )
                st.session_state["split_data"] = {
                    "X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test,
                    "is_stratified": is_strat, "test_size": test_split_ratio
                }

                train_status.info(f"2/4: Fitting StandardScaler and training {selected_algo} Classifier...")
                train_progress.progress(60)

                model_pkg = train_model(
                    X_train=X_train,
                    y_train=y_train,
                    algorithm=selected_algo,
                    params=algo_params,
                    class_names=class_names
                )

                train_status.info("3/4: Computing test set evaluation metrics...")
                train_progress.progress(85)
                time.sleep(0.2)

                eval_res = evaluate_model(
                    model_or_pipeline=model_pkg,
                    X_test=X_test,
                    y_test=y_test,
                    class_names=class_names
                )

                model_pkg["test_metrics"] = {
                    "accuracy": eval_res["accuracy"] * 100,
                    "precision": eval_res["precision"] * 100,
                    "recall": eval_res["recall"] * 100,
                    "f1_score": eval_res["f1_score"] * 100
                }

                save_model(model_pkg, DEFAULT_MODEL_PATH)
                save_evaluation_outputs(eval_res, "outputs")

                st.session_state["model_package"] = model_pkg
                st.session_state["eval_results"] = eval_res

                train_progress.progress(100)
                train_status.empty()
                train_progress.empty()

                st.success(f"🎉 {selected_algo} model successfully trained and serialized! Test Accuracy: {eval_res['accuracy']*100:.2f}%.")

            except Exception as e:
                st.error(f"Training failed: {e}")

        # Active Model Card
        if st.session_state["model_package"] is not None:
            active_pkg = st.session_state["model_package"]
            st.markdown("#### 📋 Currently Active Model Specifications")
            st.markdown(f"""
            <div class="tech-card">
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; font-size:0.88rem;">
                    <div><strong>Algorithm:</strong> <span class="mono-font" style="color:#0284C7;">{active_pkg.get('algorithm', 'N/A')}</span></div>
                    <div><strong>Trained Classes:</strong> {', '.join(active_pkg.get('class_names', []))}</div>
                    <div><strong>Training Samples:</strong> {active_pkg.get('training_samples_count', 'N/A')}</div>
                    <div><strong>Input Features:</strong> {active_pkg.get('n_features', 8160)} Descriptors</div>
                    <div><strong>Saved Artifact:</strong> <code>{DEFAULT_MODEL_PATH}</code></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 6: 📈 PERFORMANCE EVALUATION
# =============================================================================
with tabs[5]:
    st.markdown("### 📈 Comprehensive Test Set Performance Evaluation")
    st.markdown("""
    All metrics, confusion matrices, and per-class reports below are evaluated strictly on the independent, held-out test partition.
    """)

    if st.session_state["eval_results"] is None and st.session_state["model_package"] is not None and st.session_state["split_data"] is not None:
        try:
            st.session_state["eval_results"] = evaluate_model(
                model_or_pipeline=st.session_state["model_package"],
                X_test=st.session_state["split_data"]["X_test"],
                y_test=st.session_state["split_data"]["y_test"],
                class_names=st.session_state["model_package"]["class_names"]
            )
        except Exception:
            pass

    if st.session_state["eval_results"] is None:
        st.warning("⚠️ No evaluation results available yet. Please train a model in the **'Model Training Studio'** tab.")
    else:
        eval_data = st.session_state["eval_results"]
        acc = eval_data["accuracy"] * 100
        prec = eval_data["precision"] * 100
        rec = eval_data["recall"] * 100
        f1 = eval_data["f1_score"] * 100

        # KPI Performance Cards
        card1, card2, card3, card4 = st.columns(4)
        with card1:
            st.markdown(f"""
            <div class="kpi-card acc">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">Test Accuracy</div>
                <div class="kpi-value acc">{acc:.2f}%</div>
                <div class="kpi-sub">Overall Correctness</div>
            </div>
            """, unsafe_allow_html=True)
        with card2:
            st.markdown(f"""
            <div class="kpi-card prec">
                <div class="kpi-icon">🔍</div>
                <div class="kpi-label">Weighted Precision</div>
                <div class="kpi-value prec">{prec:.2f}%</div>
                <div class="kpi-sub">Positive Predictive Value</div>
            </div>
            """, unsafe_allow_html=True)
        with card3:
            st.markdown(f"""
            <div class="kpi-card rec">
                <div class="kpi-icon">📡</div>
                <div class="kpi-label">Weighted Recall</div>
                <div class="kpi-value rec">{rec:.2f}%</div>
                <div class="kpi-sub">Sensitivity Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with card4:
            st.markdown(f"""
            <div class="kpi-card f1">
                <div class="kpi-icon">⚖️</div>
                <div class="kpi-label">Weighted F1-Score</div>
                <div class="kpi-value f1">{f1:.2f}%</div>
                <div class="kpi-sub">Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations: Confusion Matrix & Per-Class Comparison
        ev_col1, ev_col2 = st.columns([1, 1])

        with ev_col1:
            st.markdown("#### 🎯 Confusion Matrix Heatmap")
            cm_fig = plot_confusion_matrix(
                cm=eval_data["confusion_matrix"],
                class_names=eval_data["class_names"],
                title="Confusion Matrix"
            )
            st.pyplot(cm_fig)
            plt.close(cm_fig)

        with ev_col2:
            st.markdown("#### 📊 Per-Class Metric Breakdown")
            class_fig = plot_per_class_metrics(
                report_dict=eval_data["report_dict"],
                class_names=eval_data["class_names"]
            )
            st.pyplot(class_fig)
            plt.close(class_fig)

        # Classification Report Table
        st.markdown("---")
        st.markdown("#### 📋 Detailed Classification Report Table")
        report_df = eval_data["report_df"]
        st.dataframe(
            report_df.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}"
            }),
            use_container_width=True
        )

        # Download Report CSV
        csv_data = report_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Classification Report (CSV)",
            data=csv_data,
            file_name="kidney_classification_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# =============================================================================
# TAB 7: 🎯 LIVE DIAGNOSTIC INFERENCE STUDIO
# =============================================================================
with tabs[6]:
    st.markdown("### 🎯 Real-Time Single Image Diagnostic Inference")
    st.markdown("""
    Upload a kidney ultrasound scan or click any quick-access sample preset below to run real-time inference.
    """)

    if st.session_state["model_package"] is None:
        st.warning("⚠️ No trained model is currently loaded. Please train a model or use the sidebar **'1-Click Auto Demo'**.")
    else:
        model_pkg = st.session_state["model_package"]

        # Quick-Access Sample Presets Bar
        st.markdown("#### ⚡ Quick-Access Sample Testing Presets")
        preset_c1, preset_c2, preset_c3, preset_c4 = st.columns(4)
        
        sample_path_map = {
            "Normal": "sample_images/test_normal_ultrasound.png",
            "Cyst": "sample_images/test_cyst_ultrasound.png",
            "Stone": "sample_images/test_stone_ultrasound.png",
            "Tumor": "sample_images/test_tumor_ultrasound.png"
        }

        with preset_c1:
            if st.button("🟢 Test Normal Sample", use_container_width=True):
                st.session_state["selected_preset_img"] = sample_path_map["Normal"]
        with preset_c2:
            if st.button("🔵 Test Cyst Sample", use_container_width=True):
                st.session_state["selected_preset_img"] = sample_path_map["Cyst"]
        with preset_c3:
            if st.button("🟠 Test Stone Sample", use_container_width=True):
                st.session_state["selected_preset_img"] = sample_path_map["Stone"]
        with preset_c4:
            if st.button("🔴 Test Tumor Sample", use_container_width=True):
                st.session_state["selected_preset_img"] = sample_path_map["Tumor"]

        st.markdown("<br>", unsafe_allow_html=True)

        # Upload or Select Input
        pred_in_c1, pred_in_c2 = st.columns([1.5, 1.5])
        with pred_in_c1:
            uploaded_test_file = st.file_uploader(
                "📤 Upload Custom Ultrasound Scan:",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                key="pred_file_uploader_v2"
            )
        with pred_in_c2:
            sample_pred_choice = None
            if summary["exists"] and summary["total_images"] > 0:
                sample_test_files = []
                for c in summary["class_names"]:
                    c_dir = os.path.join(dataset_path_input, c)
                    files = [os.path.join(c_dir, f) for f in os.listdir(c_dir) if not f.startswith(".")]
                    if files:
                        sample_test_files.extend(files[:2])
                if sample_test_files:
                    sample_pred_choice = st.selectbox(
                        "📁 Or Select from Dataset Samples:",
                        options=["-- None (Use Uploaded / Preset) --"] + sample_test_files,
                        index=0
                    )

        target_pred_image = None
        if uploaded_test_file is not None:
            target_pred_image = uploaded_test_file
        elif sample_pred_choice and sample_pred_choice != "-- None (Use Uploaded / Preset) --":
            target_pred_image = sample_pred_choice
        elif st.session_state["selected_preset_img"] and os.path.exists(st.session_state["selected_preset_img"]):
            target_pred_image = st.session_state["selected_preset_img"]
        elif os.path.exists("sample_images/test_normal_ultrasound.png"):
            target_pred_image = "sample_images/test_normal_ultrasound.png"

        if target_pred_image is not None:
            try:
                with st.spinner("Processing image through 5-stage pipeline and extracting 8,160D features..."):
                    pred_res = predict_single_image(model_pkg, target_pred_image)

                pred_class = pred_res["predicted_class"]
                conf_pct = pred_res["confidence_percent"]

                # Side-by-Side Visuals: Original vs Preprocessed vs Result
                vis_c1, vis_c2, vis_c3 = st.columns([1.2, 1.2, 1.6])

                with vis_c1:
                    st.markdown("**1. Raw Input Scan**")
                    st.image(pred_res["preprocessed_images"]["original"], use_container_width=True)

                with vis_c2:
                    st.markdown("**2. Preprocessed (Enhanced)**")
                    st.image(pred_res["preprocessed_images"]["enhanced"], use_container_width=True)

                with vis_c3:
                    st.markdown("**3. AI Diagnostic Verdict**")
                    badge_style = f"diag-{pred_class}" if pred_class in ["Normal", "Cyst", "Stone", "Tumor"] else "diag-Other"
                    conf_display = f"{conf_pct:.1f}%" if conf_pct is not None else "N/A"
                    
                    st.markdown(f"""
                    <div class="diagnosis-result-card {badge_style}">
                        <div class="diag-header-label">Predicted Diagnostic Classification</div>
                        <div class="diag-class-title">{pred_class}</div>
                        <div class="diag-confidence-meter">
                            <span>🎯 Confidence:</span>
                            <span style="color:#0F172A;">{conf_display}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Class Probability Distribution
                if pred_res["supports_probability"] and pred_res["probabilities"]:
                    st.markdown("---")
                    st.markdown("#### 📊 Model Probability Distribution Across All Classes")
                    
                    prob_df = pd.DataFrame(
                        list(pred_res["probabilities"].items()),
                        columns=["Class", "Probability"]
                    )
                    prob_df["Confidence (%)"] = prob_df["Probability"] * 100

                    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=100)
                    palette_map = {"Normal": "#10B981", "Cyst": "#3B82F6", "Stone": "#F59E0B", "Tumor": "#EF4444"}
                    bar_colors = [palette_map.get(c, "#6B7280") for c in prob_df["Class"]]

                    sns.barplot(data=prob_df, x="Confidence (%)", y="Class", palette=bar_colors, ax=ax)
                    ax.set_xlim(0, 105)
                    ax.set_xlabel("Calibrated Confidence (%)", fontsize=10, fontweight="bold")
                    ax.set_ylabel("")
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    for p in ax.patches:
                        val = p.get_width()
                        ax.annotate(f"{val:.1f}%", (val + 1.2, p.get_y() + p.get_height() / 2),
                                    va="center", fontsize=9, fontweight="bold", color="#1E293B")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # AI Inspection Report Summary
                st.markdown("---")
                st.markdown("#### 📑 Diagnostic Inspection Summary Card")
                report_summary_md = f"""
### 🏥 RenalAI - Automated Ultrasound Inspection Summary
- **Timestamp:** {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Classifier Algorithm:** {model_pkg.get('algorithm', 'N/A')}
- **Predicted Category:** **{pred_class}**
- **Algorithmic Confidence:** **{conf_display}**
- **Feature Vector Dimensions:** 8,160 descriptors (8,100 HOG + 60 GLCM)
- **Spatial Resolution:** 128×128 pixels (Standardized)
- **Clinical Notice:** Educational demonstration only. Not for medical diagnostic use.
                """
                st.markdown(report_summary_md)

                # Download Patient Report Text
                st.download_button(
                    label="📥 Download Diagnostic AI Report Summary (TXT)",
                    data=report_summary_md,
                    file_name=f"kidney_ai_report_{pred_class.lower()}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Inference error: {e}")
        else:
            st.info("💡 Upload an image file or choose a preset sample above to run the diagnostic classifier.")


# =============================================================================
# TAB 8: ℹ️ MODEL SPECS & SYSTEM METADATA
# =============================================================================
with tabs[7]:
    st.markdown("### ℹ️ Active Model Metadata & System Specifications")

    if st.session_state["model_package"] is not None:
        pkg = st.session_state["model_package"]
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown("#### 🤖 Model Configuration")
            st.markdown(f"""
            - **Algorithm:** {pkg.get('algorithm', 'Unknown')}
            - **Target Resolution:** {pkg.get('image_size', (128, 128))[0]}×{pkg.get('image_size', (128, 128))[1]} pixels
            - **Target Classes:** `{', '.join(pkg.get('class_names', []))}`
            - **Feature Dimension:** {pkg.get('n_features', 8160)}
            - **Training Timestamp:** {pkg.get('trained_at', 'N/A')}
            - **Serialization Path:** `{DEFAULT_MODEL_PATH}`
            """)

        with info_col2:
            st.markdown("#### 🧬 Feature Extraction Details")
            st.markdown(f"""
            - **Descriptor Framework:** Hybrid HOG (Edge Gradients) + GLCM (Statistical Co-occurrence)
            - **HOG Orientations:** 9 directional bins
            - **HOG Cell Size:** 8×8 pixels
            - **HOG Block Normalization:** 2×2 blocks (L2-Hys)
            - **GLCM Statistical Properties:** Contrast, Dissimilarity, Homogeneity, Energy, Correlation
            - **GLCM Angles:** 0°, 45°, 90°, 135° across multi-scale distances [1, 2, 3]
            """)

        if "test_metrics" in pkg:
            st.markdown("---")
            st.markdown("#### 🎯 Serialized Test Metrics")
            st.json(pkg["test_metrics"])
    else:
        st.info("No active model loaded. Train or load a model to inspect metadata.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="app-footer">
    <strong>RENALAI: KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING</strong><br>
    Academic & Educational Machine Learning Project &copy; 2026 | Built for Research and Educational Demonstrations.
</div>
""", unsafe_allow_html=True)
