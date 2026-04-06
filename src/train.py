import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Configuration ---
X_PATH     = "models/X.npy"
Y_PATH     = "models/y.npy"
MODEL_PATH = "models/face_model.pkl"

def train_model():
    """
    Trains a robust Pipeline: Scaler -> PCA (Eigenfaces) -> KNN
    """
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("ERROR: Feature files not found. Run feature_engineering.py first.")
        return False

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    if len(np.unique(y)) < 2:
        print("ERROR: Need at least 2 users to distinguish between identities.")
        return False

    # 1. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Create the Eigenface Pipeline
    # PCA(n_components=0.95) keeps enough 'features' to explain 95% of the face variance
    face_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, whiten=True)), 
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])

    # 3. Train the Pipeline
    print(f"Training on {len(X_train)} samples...")
    face_pipeline.fit(X_train, y_train)

    # 4. Check Internal Accuracy (Optional but good for logs)
    score = face_pipeline.score(X_test, y_test)
    print(f"✅ Training Complete. Validation Accuracy: {score:.2%}")

    # 5. Save the entire Pipeline
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(face_pipeline, f)

    return True

if __name__ == "__main__":
    train_model()