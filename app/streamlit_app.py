import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys

# Add the root directory to the path so Streamlit can find your 'inference' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.predict import load_trained_model, predict_image, overlay_heatmap
from inference.severity import calculate_severity

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Pneumonia AI Detector", page_icon="🫁", layout="wide")

st.title("🫁 Attention-Enhanced Pneumonia Detection")
st.write("Upload a Chest X-ray (DICOM format) to generate a segmentation mask and severity score.")

# ==========================================
# LOAD MODEL (Cached so it doesn't reload every time)
# ==========================================
@st.cache_resource
def get_model():
    model_path = "models/best_pneumonia_model.pth"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please make sure you placed the downloaded file in the 'models' folder.")
        st.stop()
    return load_trained_model(model_path, device="cpu")

model = get_model()

# ==========================================
# UI SIDEBAR
# ==========================================
st.sidebar.title("🧠 About this AI")
st.sidebar.info(
    "This diagnostic tool uses an **Attention U-Net** with an **EfficientNet-B0** backbone. "
    "Instead of simply predicting 'Yes' or 'No', it highlights the exact infection region "
    "and calculates a clinical severity score based on the affected lung area."
)
st.sidebar.markdown("---")
st.sidebar.write("")

# ==========================================
# MAIN APP LOGIC
# ==========================================
uploaded_file = st.file_uploader("Drop a DICOM (.dcm) X-Ray file here...", type=["dcm"])

if uploaded_file is not None:
    with st.spinner("Analyzing X-Ray... Please wait."):
        try:
            # Save the uploaded file temporarily so PyDicom can read it safely
            temp_path = "temp_uploaded.dcm"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 1. Run Inference
            original_img, predicted_mask = predict_image(model, temp_path, device="cpu")
            
            # 2. Calculate Severity & Heatmap
            severity_info = calculate_severity(predicted_mask)
            overlay_img = overlay_heatmap(original_img, predicted_mask)
            
            # Clean up the temporary file
            os.remove(temp_path)

            # ==========================================
            # DISPLAY RESULTS
            # ==========================================
            st.markdown("### 📊 Diagnostic Results")
            
            # Clean UI Metrics
            col1, col2, col3 = st.columns(3)
            
            # Change color based on severity
            if severity_info['category'] == "Healthy":
                severity_color = "normal"
            elif severity_info['category'] == "Mild":
                severity_color = "off"
            else:
                severity_color = "inverse"
                
            col1.metric("Severity Category", severity_info['category'], delta_color=severity_color)
            col2.metric("Infected Area", f"{severity_info['infection_percentage']}%")
            col3.metric("Infected Pixels", severity_info['infected_pixels'])

            st.markdown("---")

            # Show Images Side-by-Side
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.image(original_img, caption="Original Chest X-Ray", use_container_width=True, clamp=True, channels="GRAY")
                
            with img_col2:
                st.image(overlay_img, caption=f"AI Heatmap Overlay ({severity_info['category']})", use_container_width=True, channels="RGB")
        
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")