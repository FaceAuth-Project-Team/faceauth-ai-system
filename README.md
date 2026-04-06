# FaceAuth AI: Biometric Security System

**FaceAuth AI** is a real-time facial recognition authentication system built with Python, OpenCV, and Scikit-Learn. It utilizes machine learning to provide a secure "Register & Login" workflow through a Streamlit web interface.

## 🚀 System Overview
The system follows a 4-stage pipeline:
1.  **Data Collection:** Capturing raw BGR images via webcam.
2.  **Preprocessing:** Grayscale conversion, Haar Cascade face detection, and Histogram Equalization.
3.  **Feature Engineering:** Flattening 64x64 processed images into feature vectors.
4.  **Classification:** Training a K-Nearest Neighbors (KNN) model to recognize registered users.

##  Here's the full run order from start to finish:
#### **Step 1 — Install dependencies**
`pip install opencv-python numpy scikit-learn streamlit`
#### **Step 2 — Register faces (repeat for each user, minimum 2)**  
`python src/data_collection.py`  
- Type a name when prompted  
- Look at the webcam window  
- Wait for 10 photos to be captured automatically  
- un again with a different name for the second user  
#### **Step 3 — Preprocess the captured images**  
`python src/preprocessing.py`  

Cleans and resizes all photos into data/processed/
#### **Step 4 — Extract features**  
`python src/feature_engineering.py`  

Converts images into ML-ready arrays, saves to models/  
#### **Step 5 — Train the model**  
`python src/train.py`  

Trains KNN and saves models/face_model.pkl  
#### **Step 6 — Evaluate (optional but recommended)**  
`python src/evaluate.py`  

Shows accuracy and confusion matrix  
#### **Step 7 — Test login standalone (optional)**
`python src/predict.py`  

Opens webcam, press Space to identify, Q to quit  
#### **Step 8 & 9 — Run the full app**
`streamlit run app/main.py`

Opens in your browser automatically. Use the sidebar to switch between Register User and Login System.  

### **The correct order every time you add a new user:**  
`data_collection → preprocessing → feature_engineering → train → run app`

## **🛡️ Features**
- Live Registration: Capture 5-15 images directly from the browser.  
![alt text](<Screenshot 2026-04-06 013354.png>)

- Security Thresholding: Adjustable confidence slider to prevent "False Positives."

- On-the-fly Training: Retrain the security model instantly after adding a new user.

- Access Logging: Real-time feedback for "Access Granted" vs "Denied."  
![alt text](<Screenshot 2026-04-06 013325.png>)

## **👥 The Team**
This project was developed by a multidisciplinary engineering team:

Lead Engineer: Yordanos Teshome

Development Team: 
  - mitiku tadesse
-  Yared mihret tesfaye
-  Yeabsera Tilaye
-  Yisheak Alelign