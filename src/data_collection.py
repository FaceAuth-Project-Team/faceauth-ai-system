"""
Step 2: Face Registration (Data Collection)
============================================
This script opens the webcam, detects the user's face,
and saves 10 photos to:  data/raw/<username>/

How to run:
    python src/data_collection.py

Requirements:
    pip install opencv-python numpy
"""

import cv2
import os
import time

# ── Configuration ────────────────────────────────────────────────
DATASET_DIR = "data/raw"          # Where face images are saved
NUM_IMAGES = 10                   # How many photos to capture per user
CAPTURE_DELAY = 0.5               # Seconds to wait between each capture
IMG_SIZE = (200, 200)             # Save images at this size

# Path to OpenCV's built-in face detector file
# (installed automatically with opencv-python)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# ─────────────────────────────────────────────────────────────────


def get_username() -> str:
    """Ask the user to type their name (used as the folder name)."""
    while True:
        name = input("\nEnter your name or ID to register: ").strip()
        if name:                          # make sure it's not empty
            return name
        print(" Name cannot be empty. Please try again.")


def create_user_folder(username: str) -> str:
    """
    Create the folder  data/raw/<username>/
    Returns the folder path as a string.
    """
    folder = os.path.join(DATASET_DIR, username)
    os.makedirs(folder, exist_ok=True)   # won't crash if folder already exists
    return folder


def register_face(username: str) -> None:
    """
    Main function:
      1. Opens the webcam
      2. Shows a live preview with the detected face highlighted
      3. Auto-captures NUM_IMAGES photos
      4. Saves them to data/raw/<username>/
    """
    folder = create_user_folder(username)
    face_detector = cv2.CascadeClassifier(CASCADE_PATH)

    # Open the default webcam (index 0 = first camera on your computer)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" Could not open webcam. Make sure it is connected and not in use.")
        return

    print(f"\nWebcam is open. Look at the camera.")
    print(f"   We will capture {NUM_IMAGES} photos of your face.")
    print(f"   Press  Q  to quit early.\n")

    count = 0           # how many images we have saved so far
    last_capture = 0    # timestamp of the last capture

    while count < NUM_IMAGES:
        ret, frame = cap.read()     # read one frame from the webcam

        if not ret:                 # if reading failed, skip this frame
            continue

        # Convert to grayscale for the face detector
        # (grayscale is faster; the saved image stays in colour)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the current frame
        # scaleFactor: how much the image is scaled down each pass (1.3 = 30% reduction)
        # minNeighbors: how many detections needed before calling it a face (higher = stricter)
        faces = face_detector.detectMultiScale(
                  gray,
                  scaleFactor=1.1,
                  minNeighbors=3,
                  minSize=(50, 50)
                )
        # ── Draw a rectangle around every detected face ──────────
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ── Status text on the video window ──────────────────────
        status = f"Captured: {count}/{NUM_IMAGES}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        cv2.imshow("FaceAuth - Registration (press Q to quit)", frame)

        # ── Auto-capture if a face is found and enough time has passed ──
        now = time.time()
        if len(faces) > 0 and (now - last_capture) >= CAPTURE_DELAY:
            # Take the first (largest) detected face
            x, y, w, h = faces[0]

            # Crop just the face region from the original colour frame
            face_crop = frame[y:y + h, x:x + w]

            # Resize to a fixed size so all images are the same dimensions
            face_resized = cv2.resize(face_crop, IMG_SIZE)

            # Build a filename like  0001.jpg, 0002.jpg, ...
            filename = os.path.join(folder, f"{count + 1:04d}.jpg")
            cv2.imwrite(filename, face_resized)

            count += 1
            last_capture = now
            print(f" Saved image {count}/{NUM_IMAGES}  →  {filename}")

        # Press Q to quit early
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or Escape key
            print("\n Registration interrupted by user.")
            break

    # ── Clean up ─────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    if count == NUM_IMAGES:
        print(f"\n Registration complete!")
        print(f"   {NUM_IMAGES} images saved to:  {folder}/")
    else:
        print(f"\n Only {count} image(s) saved. Run the script again to complete registration.")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   FaceAuth – Face Registration")
    print("=" * 50)

    username = get_username()
    register_face(username)