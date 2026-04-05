"""
Step 4: Feature Engineering
=============================
Reads all preprocessed images from data/processed/,
converts them into:
    X = feature array  (each image flattened to 1D)
    y = labels array   (the username for each image)

Then saves X and y to disk so train.py can load them directly.

How to run:
    python src/feature_engineering.py

Requirements:
    pip install numpy scikit-learn opencv-python
"""

import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# ── Configuration (must match preprocessing.py) ───────────────────
PROCESSED_DIR = "data/processed"   # Input: cleaned images from Step 3
OUTPUT_DIR    = "models"           # Output: save X, y, and encoder here
IMG_SIZE      = (64, 64)           # Must match the size used in preprocessing
# ─────────────────────────────────────────────────────────────────


def load_images(processed_dir):
    """
    Walk through data/processed/<username>/ folders,
    read each image, flatten it to a 1D array, and record its label.

    Returns:
        X      : list of flattened image arrays  (one per image)
        labels : list of usernames               (one per image, same order as X)
    """
    X      = []   # will hold one flat array per image
    labels = []   # will hold the username for each image

    # Each subfolder name = one registered user
    for username in os.listdir(processed_dir):
        user_folder = os.path.join(processed_dir, username)

        if not os.path.isdir(user_folder):
            continue   # skip any stray files

        image_count = 0

        for filename in os.listdir(user_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(user_folder, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"  Skipping (cannot read): {image_path}")
                continue

            # Resize just in case (should already be 64x64 from Step 3)
            img = cv2.resize(img, IMG_SIZE)

            # Flatten: turn 64x64 grid into a single list of 4096 numbers
            # Before: shape (64, 64)
            # After:  shape (4096,)
            flat = img.flatten()

            X.append(flat)
            labels.append(username)
            image_count += 1

        print(f"  Loaded {image_count} images for user: {username}")

    return X, labels


def build_features(processed_dir, output_dir):
    """
    Main function:
      1. Load all images and labels
      2. Convert to numpy arrays
      3. Encode usernames as numbers  (e.g. Alice=0, Bob=1)
      4. Save X, y, and the encoder to disk
    """
    print("Loading images from:", processed_dir)
    print("-" * 40)

    X, labels = load_images(processed_dir)

    if len(X) == 0:
        print("\nERROR: No images found in", processed_dir)
        print("Make sure you have run preprocessing.py first.")
        return False

    # Convert lists to numpy arrays (required by scikit-learn)
    X = np.array(X)   # shape: (total_images, 4096)

    # LabelEncoder turns string names into numbers
    # e.g.  ["Alice", "Alice", "Bob", "Bob"]  →  [0, 0, 1, 1]
    le = LabelEncoder()
    y  = le.fit_transform(labels)  # shape: (total_images,)

    print("-" * 40)
    print(f"Total images loaded : {X.shape[0]}")
    print(f"Features per image  : {X.shape[1]}  (64 x 64 pixels flattened)")
    print(f"Users found         : {list(le.classes_)}")

    # ── Save to disk ─────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    X_path       = os.path.join(output_dir, "X.npy")
    y_path       = os.path.join(output_dir, "y.npy")
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")

    np.save(X_path, X)
    np.save(y_path, y)

    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    print(f"\nSaved X             : {X_path}")
    print(f"Saved y             : {y_path}")
    print(f"Saved label encoder : {encoder_path}")
    print("\nFeature engineering complete. Ready for training (Step 5).")
    return True


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   FaceAuth - Feature Engineering")
    print("=" * 50)
    build_features(PROCESSED_DIR, OUTPUT_DIR)
