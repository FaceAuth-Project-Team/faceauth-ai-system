import sys
import os
import streamlit as st
import pickle
import cv2
import numpy as np

# Fix import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.feature_engineering import build_features
from src.train import train_model
from src.predict import predict_face
from src.data_collection import save_frame

# --- Configuration ---
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "face_model.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "models", "label_encoder.pkl")
CONST_THRESHOLD = 0.65  # Hardcoded as requested

# --- Modern UI Styling ---
st.set_page_config(page_title="FaceAuth AI", page_icon="🔐", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div { background-color: #007bff; }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 { color: #1e3d59; text-align: center; font-family: 'Segoe UI'; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/fingerprint-accepted.png", width=100)
    st.title("Control Panel")
    page = st.radio("Navigate", ["🏠 Home", "📝 Register New User", "🔑 Secure Login"])
    st.info(f"System Threshold: {int(CONST_THRESHOLD*100)}%")

# --- 🏠 HOME PAGE ---
if page == "🏠 Home":
    st.markdown("<h1>FaceAuth AI Security System</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div class="status-box">
            <h3>System Status</h3>
            <p style='color:green;'>● Online</p>
            <p>Ready for Authentication</p>
        </div>
        """, unsafe_allow_html=True)

# --- 📝 REGISTER PAGE ---
elif page == "📝 Register New User":
    st.markdown("<h1>User Registration</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        name = st.text_input("Enter Full Name", placeholder="e.g. Yordi")
        
        # --- NEW: Editable Capture Count ---
        num_images = st.slider("Target Number of Images", 5, 50, 20)
        st.info(f"Aim for {num_images} captures for better accuracy.")
        
        if "capture_count" not in st.session_state:
            st.session_state.capture_count = 0

        if st.button("Reset & Clear Camera"):
            st.session_state.capture_count = 0
            st.rerun()

    with col2:
        st.write("### Camera Feed")
        
        # Only show camera and capture if we haven't reached the limit
        if st.session_state.capture_count < num_images:
            image = st.camera_input("Capture facial data", key="reg_cam")

            if image and name.strip():
                success = save_frame(
                    image.getvalue(),
                    name.strip(),
                    st.session_state.capture_count + 1
                )

                if success:
                    st.session_state.capture_count += 1
                    st.toast(f"Saved {st.session_state.capture_count}/{num_images}", icon="📸")
                else:
                    st.warning("No face detected! Adjust your position.")
        else:
            st.success("✅ Target Reached! You can now train the model.")
            st.balloons()

        # --- Fixed Progress Bar Logic ---
        if st.session_state.capture_count > 0:
            # Use min() to ensure value is never > 1.0 even if state glitches
            progress_val = min(float(st.session_state.capture_count / num_images), 1.0)
            st.progress(progress_val)
            st.write(f"**Status:** {st.session_state.capture_count} / {num_images} Images Collected")

    # --- Training Trigger ---
    st.markdown("---")
    if st.session_state.capture_count >= num_images:
        if st.button("🚀 Finalize & Train Security Model"):
            with st.spinner("Analyzing biometric features..."):
                features_ok = build_features(
                    os.path.join(ROOT_DIR, "data", "processed"),
                    os.path.join(ROOT_DIR, "models")
                )
                if features_ok and train_model():
                    st.success("Identity database successfully updated!")
                    
# --- 🔑 LOGIN PAGE ---
elif page == "🔑 Secure Login":
    st.markdown("<h1>Biometric Authentication</h1>", unsafe_allow_html=True)
    
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Access Denied: No trained models found. Please register a user first.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            image = st.camera_input("Look directly at the camera", key="login_cam")
        
        with col2:
            st.write("### Identity Results")
            if image:
                model = pickle.load(open(MODEL_PATH, "rb"))
                le = pickle.load(open(ENCODER_PATH, "rb"))

                img = cv2.imdecode(np.frombuffer(image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                
                # predict_face now uses CONST_THRESHOLD
                label, confidence = predict_face(img, model, le, CONST_THRESHOLD)

                if label is None:
                    st.warning("Scanning for face...")
                elif label == "Denied":
                    st.error(f"ACCESS DENIED\nConfidence: {confidence:.2%}")
                else:
                    st.success(f"ACCESS GRANTED\nWelcome, {label}\nConfidence: {confidence:.2%}")