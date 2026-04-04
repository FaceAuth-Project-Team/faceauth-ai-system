import sys
import os
import streamlit as st
import pickle
import cv2
import numpy as np
import time

# Ensure Python can see the 'src' folder correctly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_collection import capture_face_images
from src.train import train_model
from src.predict import predict_face

MODEL_PATH = "models/face_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# --- STEP 9: UI ENHANCEMENT ---
st.set_page_config(page_title="FaceAuth AI", page_icon="🔒", layout="centered")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ FaceAuth: AI Login")
st.markdown("---")

page = st.sidebar.selectbox("Navigation", ["Register User", "Login System"])

# -------- REGISTER PAGE --------
if page == "Register User":
    st.subheader("👤 User Registration")
    name = st.text_input("Enter Full Name", placeholder="e.g. John Doe")
    num_images = st.select_slider("Image Quality", options=[5, 10, 15], value=10)

    if st.button("📸 Start Face Capture"):
        if not name.strip():
            st.error("Please enter a name first.")
        else:
            with st.spinner("Accessing Camera..."):
                success = capture_face_images(name, num_images)
            if success:
                st.success(f"✅ Success: Captured {num_images} images for {name}!")
            else:
                st.warning("⚠️ Face capture failed. Check camera access.")

    st.markdown("---")
    if st.button("⚙️ Train Security Model"):
        with st.status("Training AI...", expanded=False) as status:
            if train_model():
                status.update(label="Training Complete!", state="complete")
                st.balloons()
            else:
                status.update(label="No data found to train.", state="error")

# -------- LOGIN PAGE (STEP 8 & 9) --------
elif page == "Login System":
    st.subheader("🔐 Secure Login")

    # --- STEP 8: THRESHOLD LOGIC ---
    with st.expander("Security Settings"):
        threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.6)

    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ System is empty. Please register a user first.")
    else:
        image = st.camera_input("Look at the camera to log in")
        if image:
            with st.spinner("Checking identity..."):
                model = pickle.load(open(MODEL_PATH, "rb"))
                le = pickle.load(open(ENCODER_PATH, "rb"))
                
                img = cv2.imdecode(np.frombuffer(image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                label, confidence = predict_face(img, model, le, threshold)

            if label is None:
                st.warning("❓ No face detected.")
            elif label == "Denied":
                st.error(f"🚫 ACCESS DENIED (Score: {confidence*100:.1f}%)")
                st.progress(confidence)
            else:
                st.success(f"✅ WELCOME BACK, {label.upper()}!")
                st.metric("Identity Confidence", f"{confidence*100:.1f}%")
                st.progress(confidence)