import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_face(img, model, le, threshold=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (64, 64)).flatten() / 255.0

    # Calculate Confidence using distance
    distances, _ = model.kneighbors([face])
    distance = distances[0][0]
    confidence = 1 / (1 + distance) # Simple logic: closer distance = higher score

    pred_index = model.predict([face])[0]
    label = le.inverse_transform([pred_index])[0]

    if confidence >= threshold:
        return label, confidence
    else:
        return "Denied", confidence