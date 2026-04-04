import cv2
import os
import streamlit as st

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
DATASET_PATH = "dataset"

def capture_face_images(user_name, num_images=5):
    user_path = os.path.join(DATASET_PATH, user_name)
    os.makedirs(user_path, exist_ok=True)
    
    # Try index 0 with DSHOW (Standard for Windows laptops)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not camera.isOpened():
        st.error("🚫 Camera could not be opened. Check if another app (Zoom/Chrome) is using it.")
        return False

    count = 0
    frame_window = st.empty()
    progress_bar = st.progress(0)

    while count < num_images:
        ret, frame = camera.read()
        if not ret:
            st.error("❌ Camera stopped sending data.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            
            file_path = os.path.join(user_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            
            progress_bar.progress(count / num_images)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if count >= num_images:
                break

        frame_window.image(frame, channels="BGR", caption=f"Capturing... {count}/{num_images}")
    
    camera.release()
    frame_window.empty() # Clear the camera preview when done
    
    if count >= num_images:
        return True
    return False