import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
 
# ── Configuration ─────────────────────────────────────────────────
MODEL_PATH   = "models/face_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
X_PATH       = "models/X.npy"
Y_PATH       = "models/y.npy"
# ─────────────────────────────────────────────────────────────────
 
 
def run_evaluation(model, X_test, y_test, label_encoder=None):
    """
    Evaluates the trained model on test data.
 
    Parameters:
        model         : trained KNN model
        X_test        : test feature array
        y_test        : test labels (numbers)
        label_encoder : optional, converts numbers back to usernames
                        for a more readable report
 
    Returns:
        acc    : accuracy as a float (e.g. 0.95 = 95%)
        matrix : confusion matrix as a 2D numpy array
    """
    y_pred = model.predict(X_test)
 
    # 1. Accuracy
    acc = accuracy_score(y_test, y_pred)
 
    # 2. Confusion Matrix
    matrix = confusion_matrix(y_test, y_pred)
 
    print(f"Model Accuracy   : {acc * 100:.2f}%")
    print(f"Confusion Matrix :\n{matrix}")
 
    # 3. Per-user breakdown (shows precision/recall per person)
    if label_encoder is not None:
        user_names = label_encoder.classes_
        print("\nPer-User Report:")
        print(classification_report(y_test, y_pred, target_names=user_names))
 
    return acc, matrix
 
 
# ── Entry point (standalone run) ──────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   FaceAuth - Model Evaluation")
    print("=" * 50)
 
    # Check all required files exist
    for path in [MODEL_PATH, ENCODER_PATH, X_PATH, Y_PATH]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            print("Please run train.py first.")
            exit(1)
 
    # Load model and encoder
    model         = pickle.load(open(MODEL_PATH, "rb"))
    label_encoder = pickle.load(open(ENCODER_PATH, "rb"))
 
    # Load full dataset and use 20% as test set (same split as train.py)
    from sklearn.model_selection import train_test_split
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    print(f"Test samples: {len(X_test)}\n")
    run_evaluation(model, X_test, y_test, label_encoder)
