from sklearn.metrics import accuracy_score, confusion_matrix

def run_evaluation(model, X_test, y_test):
    # Make predictions on the unseen test data
    y_pred = model.predict(X_test)

    # 1. Calculate Accuracy [cite: 279]
    acc = accuracy_score(y_test, y_pred)
    
    # 2. Create Confusion Matrix (shows which users get confused) [cite: 280]
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Current Model Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(matrix)
    
    return acc, matrix