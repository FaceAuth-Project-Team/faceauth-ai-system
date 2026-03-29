import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess_image(image_path, save_path):
    try:
        img = cv2.imread(image_path)

    
        if img is None:
            print(f"Skipping (cannot read): {image_path}")
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            print(f"No face detected in: {image_path}")
            return False

        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        face_roi = gray[y:y+h, x:x+w]

        face_roi = cv2.equalizeHist(face_roi)


        resized_face = cv2.resize(face_roi, (64, 64))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)


        cv2.imwrite(save_path, resized_face)

        return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_dataset(input_base, output_base):
    total = 0
    success = 0
    for root, dirs, files in os.walk(input_base):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_base)
                output_dir = os.path.join(output_base, relative_path)
                output_path = os.path.join(output_dir, file)

                if preprocess_image(input_path, output_path):
                    success += 1

    print("\nProcessing Complete")
    print(f"Total images scanned: {total}")
    print(f"Successfully processed: {success}")
    print(f"Failed: {total - success}")


if __name__ == "__main__":
  
    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"

    print("Starting Preprocessing...")
    process_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH)