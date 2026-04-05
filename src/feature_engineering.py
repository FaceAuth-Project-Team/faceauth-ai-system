import cv2
import numpy as np
import os

# Use the relative path from the project root to the data folder
def prepare_data(data_path="data/processed/"): 
    X = [] 
    y = [] 
    
    # Check if the path exists to avoid errors
    if not os.path.exists(data_path):
        print(f"Error: The directory {data_path} does not exist.")
        return np.array([]), np.array([])

    for user_name in os.listdir(data_path):
        user_folder = os.path.join(data_path, user_name)
        
        if not os.path.isdir(user_folder):
            continue
            
        for image_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, image_name)
            
            # 1. Read image in Grayscale as per Step 3 tasks [cite: 49]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # 2. Resize to 64x64 [cite: 48]
            img_resized = cv2.resize(img, (64, 64))
            
            # 3. Normalize pixel values [cite: 49]
            img_normalized = img_resized / 255.0
            
            # 4. Flatten into 1D array for ML input [cite: 53]
            X.append(img_normalized.flatten())
            y.append(user_name)
            
    return np.array(X), np.array(y)