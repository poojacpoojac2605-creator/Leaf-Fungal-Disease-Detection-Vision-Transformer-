import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\dell\Desktop\Cucumber Disease Recognition Dataset\Original Image\Original Image\Anthracnose\image1.jpg")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Cucumber Leaf Image")
plt.axis("off")
plt.show()
resized = cv2.resize(img_rgb, (224, 224))
plt.imshow(resized)
plt.title("Resized Image (224x224)")
plt.axis("off")
plt.show()
gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

# Apply Median Filter instead of Gaussian Blur
median = cv2.medianBlur(gray, 5)

plt.imshow(median, cmap='gray')
plt.title("Median Filtered Image")
plt.axis("off")
plt.show()

# Convert to binary image
_, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(thresh, cmap='gray')
plt.title("Segmented Diseased Area")
plt.axis("off")
plt.show()
edges = cv2.Canny(median, 50, 150)

plt.imshow(edges, cmap='gray')
plt.title("Disease Edge Detection")
plt.axis("off")
plt.show()
import os
import cv2

folder_path = r"C:\Users\dell\Desktop\Cucumber Disease Recognition Dataset\Original Image\Original Image\Anthracnose"

images = []

for file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, file)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (224,224))
        images.append(img)

print("Total images loaded:", len(images))
import os
import cv2

folder_path = r"C:\Users\dell\Desktop\Cucumber Disease Recognition Dataset\Original Image\Original Image\Anthracnose"

images = []

for file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, file)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (224,224))
        images.append(img)

print("Total images loaded:", len(images))
import os
import cv2
import numpy as np

main_folder = r"C:\Users\dell\Desktop\Cucumber Disease Recognition Dataset\Original Image\Original Image"

classes = os.listdir(main_folder)
data = []
labels = []

for label, disease in enumerate(classes):
    disease_folder = os.path.join(main_folder, disease)
    for file in os.listdir(disease_folder):
        img_path = os.path.join(disease_folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224,224))
            data.append(img)
            labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Total images:", len(data))
print("Classes:", classes)
data = data / 255.0
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy*100)
import tensorflow as tf
print(tf.__version__)
