import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Set paths for train and test folders
train_dir = r"C:/Users/karav_867vu4n/Downloads/train/train"
test_dir = r"C:/Users/karav_867vu4n/Downloads/test1/test1"
output_predictions = r"C:/Users/karav_867vu4n/Downloads/test1_predictions.txt"

# Image size for resizing
IMAGE_SIZE = (64, 64)

# Function to load images and labels based on file naming convention
def load_data(data_dir, label_from_filename=True):
    images = []
    labels = []
    filenames = []
    print(f"Loading data from {data_dir}...")
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        return np.array([]), np.array([]), []

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                filenames.append(file)
                if label_from_filename:
                    # Determine label from filename: 'cat' in name -> 0, 'dog' in name -> 1
                    if "cat" in file.lower():
                        labels.append(0)
                    elif "dog" in file.lower():
                        labels.append(1)
    print(f"Loaded {len(images)} images from {data_dir}")
    return np.array(images), np.array(labels), filenames

# Load training data
X_train, y_train, _ = load_data(train_dir)
if len(X_train) == 0 or len(y_train) == 0:
    print("Error: Training data is empty. Please check the directory and files.")
    exit()

# Load test data (no labels inferred from filenames)
X_test, _, test_filenames = load_data(test_dir, label_from_filename=False)
if len(X_test) == 0:
    print("Error: Test data is empty. Please check the directory and files.")
    exit()

# Flatten images for SVM
def flatten_images(images):
    return images.reshape(images.shape[0], -1)

try:
    X_train_flattened = flatten_images(X_train)
    X_test_flattened = flatten_images(X_test)
except Exception as e:
    print(f"Error while flattening images: {e}")
    exit()

# Normalize features
scaler = StandardScaler()
try:
    X_train_flattened = scaler.fit_transform(X_train_flattened)
    X_test_flattened = scaler.transform(X_test_flattened)
except Exception as e:
    print(f"Error during normalization: {e}")
    exit()

# Train SVM classifier
print("Training SVM classifier...")
svm = SVC(kernel="linear", C=1.0, random_state=42)
try:
    svm.fit(X_train_flattened, y_train)
    print("SVM training completed.")
except Exception as e:
    print(f"Error during SVM training: {e}")
    exit()

# Make predictions
print("Making predictions on test images...")
try:
    y_pred = svm.predict(X_test_flattened)
    print("Predictions completed.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Save predictions
print("Saving predictions to file...")
try:
    with open(output_predictions, "w") as f:
        f.write("Filename,Prediction\n")
        for filename, prediction in zip(test_filenames, y_pred):
            label = "Cat" if prediction == 0 else "Dog"
            f.write(f"{filename},{label}\n")
    print(f"Predictions saved to {output_predictions}")
except Exception as e:
    print(f"Error while saving predictions: {e}")
