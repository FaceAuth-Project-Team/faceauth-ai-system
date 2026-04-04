import cv2

def preprocess_image(img):
    # Ensure image is resized to match training size
    img = cv2.resize(img, (64, 64))
    # Normalize pixel values to be between 0 and 1
    img = img.flatten() / 255.0
    return img