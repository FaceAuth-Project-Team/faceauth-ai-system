"""
Step 8 & 9: Streamlit UI - FaceAuth Main App
=============================================
Run with:
    streamlit run app/main.py
"""

import sys
import os
import streamlit as st
import pickle
import cv2
import numpy as np

# Ensure Python can see the 'src' folder correctly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_collection import capture_face_images
from src.train import train_model
from src.predict import predict_face

# ── Paths ─────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "face_model.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "models", "label_encoder.pkl")
# ─────────────────────────────────────────────────────────────────

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(page_title="FaceAuth AI", page_icon="lock", layout="centered")
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #007bff;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("FaceAuth: AI Login")
st.markdown("---")

page = st.sidebar.selectbox("Navigation", ["Register User", "Login System"])

# ── REGISTER PAGE ─────────────────────────────────────────────────
if page == "Register User":
    st.subheader("User Registration")

    name       = st.text_input("Enter Full Name", placeholder="e.g. John Doe")
    num_images = st.select_slider("Number of Images", options=[5, 10, 15], value=10)

    if st.button("Start Face Capture"):
        if not name.strip():
            st.error("Please enter a name first.")
        else:
            with st.spinner("Accessing camera... Look at the webcam window that opens."):
                success = capture_face_images(name.strip(), num_images)
            if success:
                st.success(f"Captured {num_images} images for {name}. You can now train the model.")
            else:
                st.warning("Face capture failed or was interrupted. Try again.")

    st.markdown("---")

    if st.button("Train Security Model"):
        with st.spinner("Training model..."):
            result = train_model()
        if result:
            st.success("Model trained successfully!")
            st.balloons()
        else:
            st.error("Training failed. Make sure at least 2 users are registered.")

# ── LOGIN PAGE ────────────────────────────────────────────────────
elif page == "Login System":
    st.subheader("Secure Login")

    # Step 8: Confidence threshold
    with st.expander("Security Settings"):
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6,
                              help="Higher = stricter. Lower = more lenient.")

    if not os.path.exists(MODEL_PATH):
        st.warning("No model found. Please register users and train the model first.")
    else:
        image = st.camera_input("Look at the camera to log in")

        if image:
            with st.spinner("Checking identity..."):
                # Load model and encoder
                model = pickle.load(open(MODEL_PATH, "rb"))
                le    = pickle.load(open(ENCODER_PATH, "rb"))

                # Decode the image from Streamlit camera input
                img = cv2.imdecode(
                    np.frombuffer(image.getvalue(), np.uint8),
                    cv2.IMREAD_COLOR
                )

                # Run prediction
                label, confidence = predict_face(img, model, le, threshold)

            # Show result
            if label is None:
                st.warning("No face detected. Please look directly at the camera.")

            elif label == "Denied":
                st.error(f"ACCESS DENIED  (Confidence: {confidence * 100:.1f}%)")
                st.progress(float(confidence))
            else:
                st.success(f"WELCOME BACK, {label.upper()}!")
                st.metric("Identity Confidence", f"{confidence * 100:.1f}%")
                st.progress(float(confidence))