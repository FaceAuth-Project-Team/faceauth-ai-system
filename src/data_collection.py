import cv2
import os
import numpy as np

DATASET_DIR    = "data/raw"
PROCESSED_DIR  = "data/processed"
RAW_SIZE       = (200, 200)
PROCESSED_SIZE = (64, 64)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def preprocess_face(face_bgr):
    gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    equaled = cv2.equalizeHist(gray)
    resized = cv2.resize(equaled, PROCESSED_SIZE)
    return resized


def save_frame(image_bytes, username, count):

    raw_folder       = os.path.join(DATASET_DIR, username)
    processed_folder = os.path.join(PROCESSED_DIR, username)

    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 3)

    if len(faces) == 0:
        return False

    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    face_crop = img[y:y+h, x:x+w]

    raw_path  = os.path.join(raw_folder, f"{count:04d}.jpg")
    proc_path = os.path.join(processed_folder, f"{count:04d}.jpg")

    cv2.imwrite(raw_path, cv2.resize(face_crop, RAW_SIZE))
    cv2.imwrite(proc_path, preprocess_face(face_crop))

    return True