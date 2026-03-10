#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install gradio')
get_ipython().system('pip install opencv-python')


# In[85]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[88]:


# Step 1: Load images and preprocess
def load_images(dataset_path, img_size=(100, 100)):
    images = []
    labels = []
    label_dict = {}
    person_folders = sorted(os.listdir(dataset_path))  # Ensure consistent label mapping
    
    for label, person in enumerate(person_folders):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            label_dict[label] = person  # Mapping label to person's name
            for image_name in os.listdir(person_path):
                img_path = os.path.join(person_path, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)  # Resize to fixed size
                images.append(img.flatten())  # Flatten to 1D array
                labels.append(label)
    
    return np.array(images), np.array(labels), label_dict

# Set dataset path
dataset_path = r"C:\Users\BeepBoopPop\OneDrive\Desktop\faces_dataset"
X, y, label_dict = load_images(dataset_path)

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=4)


# In[97]:


def augment_image(image):
    flipped = cv2.flip(image, 1)  # Horizontal flip
    noisy = image + np.random.normal(0, 10, image.shape)  # Add noise
    return [image, flipped, np.clip(noisy, 0, 255).astype(np.uint8)]

# Augment the training set
augmented_X_train = []
augmented_y_train = []

for image, label in zip(X_train, y_train):
    augmented_images = augment_image(image.reshape(100, 100))  # Reshape to 2D image for augmentation
    for augmented_image in augmented_images:
        augmented_X_train.append(augmented_image.flatten())  # Flatten and add to the augmented set
        augmented_y_train.append(label) 

# Convert to numpy arrays
X_train_augmented = np.array(augmented_X_train)
y_train_augmented = np.array(augmented_y_train)


# In[98]:


# Step 3: Apply PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_augmented)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=200)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 4: Apply LDA
num_classes = len(np.unique(y_train_augmented))
lda_components = min(8, num_classes - 1) 

lda = LDA(n_components=lda_components)
X_train_lda = lda.fit_transform(X_train_pca, y_train_augmented)
X_test_lda = lda.transform(X_test_pca)


# In[100]:


# Step 5: Train classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X_train_lda, y_train_augmented)

print(f"Number of Training Samples: {len(y_train)}")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Number of Classes: {len(np.unique(y_train))}")

# Step 6: Predictions
y_pred = classifier.predict(X_test_lda)
accuracy = accuracy_score(y_test, y_pred)
print(f"Face Recognition Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=[label_dict[i] for i in label_dict]))


# In[101]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_dict.values(), yticklabels=label_dict.values())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[102]:


# Gradio Interface
def recognize_face(image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((100, 100))
    img_array = np.array(img).flatten().reshape(1, -1)
    img_pca = pca.transform(img_array)
    img_lda = lda.transform(img_pca)
    prediction = classifier.predict(img_lda)
    return f"Predicted Person: {label_dict[prediction[0]]}"


# In[103]:


iface = gr.Interface(
    fn=recognize_face,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Face Recognition",
    description="Upload a face image to recognize the person."
)

iface.launch()


# In[ ]:




