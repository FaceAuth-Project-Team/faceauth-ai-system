"""
Step 7: Face Login System (Prediction / Inference)
====================================================
This script takes a live image, preprocesses it the same way
as training, and predicts who the person is using the saved model.

Used by the Streamlit app (app/main.py) via:
    from src.predict import predict_face

Requirements:
    pip install opencv-python numpy scikit-learn
"""

import cv2
import numpy as np
import pickle
import os

# ── Configuration (must match preprocessing.py settings) ─────────
IMG_SIZE = (64, 64)
MODEL_PATH = "models/face_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# ─────────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def preprocess_face(img):
    """
    Takes a raw BGR image (numpy array from OpenCV or Streamlit camera),
    detects the face, and returns a flattened 1D array ready for the model.

    Returns None if no face is detected.
    """
    # Convert to grayscale (same as preprocessing.py)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None  # No face found

    # Pick the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    # Crop, enhance contrast, and resize (same steps as preprocessing.py)
    face_roi = gray[y:y + h, x:x + w]
    face_roi = cv2.equalizeHist(face_roi)
    resized = cv2.resize(face_roi, IMG_SIZE)

    # Flatten to 1D array (same as feature_engineering.py)
    return resized.flatten()


def predict_face(img, model, label_encoder, threshold=0.6):
    """
    Main prediction function called by the Streamlit app.

    Parameters:
        img           : BGR image as numpy array (from cv2 or st.camera_input)
        model         : trained ML model loaded from face_model.pkl
        label_encoder : LabelEncoder loaded from label_encoder.pkl
        threshold     : minimum confidence to grant access (0.0 - 1.0)

    Returns:
        (label, confidence)
        - label      : predicted username, or "Denied" if below threshold,
                       or None if no face detected
        - confidence : float between 0 and 1 (how sure the model is)
    """
    # Step 1: Preprocess the image into a feature vector
    features = preprocess_face(img)

    if features is None:
        return None, 0.0  # No face detected

    # Step 2: Reshape to 2D array (model expects shape [1, n_features])
    features = features.reshape(1, -1)

    # Step 3: Get prediction probabilities for each registered user
    # predict_proba returns an array like [0.05, 0.91, 0.04]
    # meaning 91% chance it's user at index 1
    probabilities = model.predict_proba(features)[0]

    # Step 4: Find the highest confidence and which user it belongs to
    best_index = np.argmax(probabilities)
    confidence = probabilities[best_index]

    # Step 5: Convert index back to the actual username
    predicted_label = label_encoder.inverse_transform([best_index])[0]

    # Step 6: Apply threshold - if not confident enough, deny access
    if confidence < threshold:
        return "Denied", confidence

    return predicted_label, confidence


def predict_from_webcam(threshold=0.6):
    """
    Standalone function to test login directly from webcam
    without the Streamlit UI.

    Run this file directly:
        python src/predict.py
    """
    # Load model and encoder
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("ERROR: Model not found. Please train the model first (python src/train.py).")
        return

    model = pickle.load(open(MODEL_PATH, "rb"))
    label_encoder = pickle.load(open(ENCODER_PATH, "rb"))

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam open. Look at the camera.")
    print("Press SPACE to capture and identify. Press Q or Escape to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Show live preview
        cv2.putText(frame, "Press SPACE to identify | Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("FaceAuth - Login", frame)

        key = cv2.waitKey(1) & 0xFF

        # Press SPACE to capture and predict
        if key == ord(" "):
            label, confidence = predict_face(frame, model, label_encoder, threshold)

            if label is None:
                result = "No face detected."
                color = (0, 165, 255)   # orange
            elif label == "Denied":
                result = f"ACCESS DENIED  (score: {confidence*100:.1f}%)"
                color = (0, 0, 255)     # red
            else:
                result = f"ACCESS GRANTED: {label.upper()}  (score: {confidence*100:.1f}%)"
                color = (0, 255, 0)     # green

            print(result)

            # Show result on screen for 2 seconds
            result_frame = frame.copy()
            cv2.putText(result_frame, result, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("FaceAuth - Login", result_frame)
            cv2.waitKey(2000)

        # Press Q or Escape to quit
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point (standalone test) ────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   FaceAuth - Face Login (Standalone Test)")
    print("=" * 50)
    predict_from_webcam(threshold=0.6)