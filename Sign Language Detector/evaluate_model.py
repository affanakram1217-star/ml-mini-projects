import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# Load your trained model
model = load_model('sign_model.keras')

# Folder and class setup
data_folder = r"C:\Users\affan\OneDrive\Desktop\Sign Language Detector\Data"
class_names = ["Hello", "I Love You", "No", "Thank You","Yes"]

imgSize = 224  # match model input layer
X_test, y_true = [], []

# Load and preprocess images
for idx, label in enumerate(class_names):
    folder_path = os.path.join(data_folder, label)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (imgSize, imgSize))
        img = img / 255.0
        X_test.append(img)
        y_true.append(idx)

X_test = np.array(X_test)
y_true = np.array(y_true)

# Predict and compute accuracy
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_true, y_pred_classes)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
