# 👁️ Real-Time Face Detection and Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C.svg?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-5C3EE8.svg?logo=opencv)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

A robust computer vision pipeline for real-time facial detection and recognition, built with **PyTorch** and **OpenCV**. This project focuses on optimizing inference latency and achieving high accuracy for live video streams.

---

## 🚀 Key Features

* **Real-Time Inference Engine (`Realtime Face Recog.py`):** Utilizes OpenCV for continuous video stream capture and processes frames through the neural network with minimal latency.
* **Deep Learning Pipeline (`train.py` & `inference.py`):** Custom PyTorch implementations demonstrating full-cycle model application, from data preprocessing to weight extraction and deployment.
* **Algorithm Benchmarking (`Traditional_FaceRecog.py`):** Includes traditional baseline methods (e.g., Eigenfaces/PCA) to validate the superior accuracy and robustness of the deep learning approach.
* **User Interface (`UI.py`):** A streamlined interface to visualize detection results in real-world scenarios.

---

## 🛠️ Tech Stack & Skills Demonstrated

* **Languages & Frameworks:** Python, PyTorch, OpenCV
* **AI/ML Concepts:** Convolutional Neural Networks (CNNs), Model Inference, Data Preprocessing
* **Software Engineering:** Clean code practices, modular architecture, environment management (`requirements.txt`)

*(Note: These skills directly align with requirements for building scalable AI systems and deploying edge models.)*

---

## 📂 Project Structure

```text
├── inference.py               # Core inference script for model evaluation
├── Realtime Face Recog.py     # Live webcam detection pipeline (Optimized latency)
├── train.py                   # Model training and optimization script
├── Traditional_FaceRecog.py   # Baseline traditional CV methods
├── UI.py                      # Visualization and interaction layer
└── requirements.txt           # Environment dependencies
```

---

## ⚙️ Quick Start (Inference Setup)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Tinking32/Real-Time-Face-Recognition.git](https://github.com/Tinking32/Real-Time-Face-Recognition.git)
   cd Real-Time-Face-Recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the real-time pipeline:**
   ```bash
   python "Realtime Face Recog.py"
   ```

*Note: Due to GitHub's file size constraints, the pre-trained PyTorch model weights (`.pth`) are not hosted in this repository but are available upon request.*
