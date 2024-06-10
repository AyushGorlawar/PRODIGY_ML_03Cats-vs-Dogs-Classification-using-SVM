import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Directory paths
dataset_path = "C:/Users/ayush/OneDrive/Desktop/intership/intern taks/PRODIGY_ML_03/train"

# Parameters
image_size = (64, 64)  # Resize images to 64x64

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels

# Load images
cats_folder = os.path.join(dataset_path, "cat")
dogs_folder = os.path.join(dataset_path, "dog")

cat_images, cat_labels = load_images_from_folder(cats_folder, 0)
dog_images, dog_labels = load_images_from_folder(dogs_folder, 1)
 
images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)
 
images = images / 255.0

 
n_samples = len(images)
images_flattened = images.reshape(n_samples, -1)
 
X_train, X_test, y_train, y_test = train_test_split(images_flattened, labels, test_size=0.2, random_state=42)

# Create and train the SVM model
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Predict on test data
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot a few test images with their predictions
def plot_samples(images, labels, preds, n=10):
    n_samples = min(len(images), n)
    plt.figure(figsize=(20, 4))
    for i in range(n_samples):
        ax = plt.subplot(1, n_samples, i + 1)
        plt.imshow(images[i].reshape(64, 64, 3))
        plt.title(f"True: {labels[i]}, Pred: {preds[i]}")
        plt.axis("off")
    plt.show()

plot_samples(X_test[:10].reshape(-1, 64, 64, 3), y_test[:10], y_pred[:10])
