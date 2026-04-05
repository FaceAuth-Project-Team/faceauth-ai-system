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
from src.data_collection import save_frame  # ✅ NEW

# Paths
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "face_model.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "models", "label_encoder.pkl")

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="FaceAuth AI",
    page_icon=":lock:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── (YOUR CSS + UI DESIGN REMAINS SAME) ──
# 👉 I’m skipping re-pasting your long CSS for clarity
# KEEP your existing CSS block here EXACTLY as is

# ── Hero ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 0 2rem;">
    <h1>FaceAuth AI</h1>
</div>
""", unsafe_allow_html=True)

tab_register, tab_login = st.tabs(["Register", "Login"])

# ════════════════════════════════════════════════════════
# REGISTER TAB (FIXED)
# ════════════════════════════════════════════════════════
with tab_register:

    name       = st.text_input("Full Name")
    num_images = st.select_slider("Capture Quality", options=[5, 10, 15], value=10)

    if "capture_count" not in st.session_state:
        st.session_state.capture_count = 0

    st.info("Use the camera below. Capture multiple images.")

    image = st.camera_input("", key="register_cam")

    if image and name.strip():

        success = save_frame(
            image.getvalue(),
            name.strip(),
            st.session_state.capture_count + 1
        )

        if success:
            st.session_state.capture_count += 1
            st.success(f"Captured {st.session_state.capture_count}/{num_images}")
        else:
            st.warning("No face detected. Try again.")

    if st.session_state.capture_count > 0:
        st.progress(st.session_state.capture_count / num_images)

    if st.session_state.capture_count >= num_images:
        st.success(f"{num_images} images captured for {name}")
        st.session_state.capture_count = 0

    st.markdown("---")

    if st.button("Train Security Model"):
        with st.spinner("Extracting features..."):
            features_ok = build_features(
                os.path.join(ROOT_DIR, "data", "processed"),
                os.path.join(ROOT_DIR, "models")
            )

        if not features_ok:
            st.error("Need at least 2 users.")
        else:
            with st.spinner("Training model..."):
                result = train_model()

            if result:
                st.success("Model trained successfully!")
            else:
                st.error("Training failed.")

# ════════════════════════════════════════════════════════
# LOGIN TAB (UNCHANGED)
# ════════════════════════════════════════════════════════
with tab_login:

    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)

    if not os.path.exists(MODEL_PATH):
        st.warning("Train model first.")
    else:
        image = st.camera_input("", key="login_cam")

        if image:
            model = pickle.load(open(MODEL_PATH, "rb"))
            le    = pickle.load(open(ENCODER_PATH, "rb"))

            img = cv2.imdecode(
                np.frombuffer(image.getvalue(), np.uint8),
                cv2.IMREAD_COLOR
            )

            label, confidence = predict_face(img, model, le, threshold)

            if label is None:
                st.warning("No face detected")

            elif label == "Denied":
                st.error(f"Access Denied ({confidence:.2f})")

            else:
                st.success(f"Welcome {label} ({confidence:.2f})")