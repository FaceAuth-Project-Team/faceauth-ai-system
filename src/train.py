import os
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from src.image_processing import preprocess_image

DATASET_PATH = "dataset"
MODEL_PATH = "models/face_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

def train_model():
    X, y = [], []
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found")
        return False

    for user in os.listdir(DATASET_PATH):
        user_path = os.path.join(DATASET_PATH, user)
        if not os.path.isdir(user_path): continue
        
        for img_name in os.listdir(user_path):
            if not img_name.endswith((".jpg", ".png")): continue
            
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                try:
                    X.append(preprocess_image(img))
                    y.append(user)
                except: continue

    if len(X) == 0: return False

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y_encoded)

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(le, f)
    return True