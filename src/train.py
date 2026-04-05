import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
# ── Configuration ─────────────────────────────────────────────────
X_PATH     = "models/X.npy"
Y_PATH     = "models/y.npy"
MODEL_PATH = "models/face_model.pkl"
# ─────────────────────────────────────────────────────────────────
 
 
def train_model():
    """
    Loads X and y from disk, trains KNN, and saves the model.
    Returns True if successful, False otherwise.
    """
 
    # Load features saved by feature_engineering.py
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("ERROR: Feature files not found.")
        print("Please run feature_engineering.py first.")
        return False
 
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
 
    print(f"Loaded X: {X.shape}  (images x features)")
    print(f"Loaded y: {y.shape}  (labels)")
 
    if len(np.unique(y)) < 2:
        print("ERROR: Need at least 2 registered users to train.")
        return False
 
    if len(X) < 5:
        print("ERROR: Not enough images to train. Register more faces.")
        return False
 
    # Split the data: 80% for learning, 20% for testing performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    # Initialize and train KNN
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
 
    # Save the model
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
 
    print("Model trained and saved to models/face_model.pkl")
    return True
 
 
# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   FaceAuth - Model Training")
    print("=" * 50)
    train_model()
