import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def train_model(X, y):
    # 1. Split the data: 80% for learning, 20% for testing performance [cite: 264]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Initialize KNN (The project's recommended model) [cite: 267]
    model = KNeighborsClassifier(n_neighbors=3)

    # 3. Train the model [cite: 272]
    model.fit(X_train, y_train)

    # 4. Save the model to the required directory [cite: 273, 338]
    with open('models/face_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved to models/face_model.pkl")
    return model, X_test, y_test