import sys
import cv2
import time
import torch
import joblib
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from facenet_pytorch import MTCNN
import os
import json
from utils.DataLoader import FaceDataset
from models.FaceClassifier import FaceClassifier

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Face Recognition")
        self.image_label = QLabel("Camera not started")
        self.status_label = QLabel("No model loaded")
        self.upload_btn = QPushButton("Upload Model (.pth, .pkl, .json)")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.upload_btn)
        self.setLayout(layout)

        self.upload_btn.clicked.connect(self.load_model)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture(0)

        self.current_model = None
        self.model_type = None

        self.mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.haar_path = 'models/haarcascade_frontalface_default.xml'
        self.haar_detector = cv2.CascadeClassifier(self.haar_path)

        self.timer.start(200)

    def load_model(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Model Files", "", "Model Files (*.pth *.pkl *.json)")
        if not file_paths:
            return

        model_parts = {}
        for file_path in file_paths:
            try:
                fname = os.path.basename(file_path).lower()
                if file_path.endswith(".pth"):
                    self.current_model = torch.load(file_path, map_location='cpu')
                    self.model_type = "deep"
                elif file_path.endswith(".pkl"):
                    loaded = joblib.load(file_path)
                    if isinstance(loaded, dict):
                        for key, value in loaded.items():
                            if "scaler" in key or hasattr(value, 'transform') and hasattr(value, 'fit'):
                                model_parts['scaler'] = value
                            elif "pca" in key or 'explained_variance_ratio_' in dir(value):
                                model_parts['pca'] = value
                            elif "lda" in key or hasattr(value, 'coef_'):
                                model_parts['lda'] = value
                            elif "classifier" in key or hasattr(value, 'predict'):
                                model_parts['classifier'] = value
                            elif "label" in key or isinstance(value, dict):
                                model_parts['label_dict'] = value
                    else:
                        if "lda" in fname:
                            model_parts['lda'] = loaded
                        elif "svm" in fname or "svc" in fname or "clf" in fname:
                            model_parts['classifier'] = loaded
                        elif "pca" in fname:
                            model_parts['pca'] = loaded
                        elif "scaler" in fname:
                            model_parts['scaler'] = loaded
                elif file_path.endswith(".json") and "label" in fname:
                    with open(file_path, 'r') as f:
                        model_parts['label_dict'] = json.load(f)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        if self.model_type == "deep":
            self.status_label.setText("Loaded deep model.")
            try:
                dataset = FaceDataset("data/filter_enhanced", transform_prob={})
                from torch.utils.data import DataLoader, random_split
                from torch import Generator

                generator = Generator().manual_seed(42)
                total_size = len(dataset)
                test_size = int(total_size * 0.2)
                val_size = int((total_size - test_size) * 0.1)
                train_size = total_size - test_size - val_size

                train_val_set, _ = random_split(dataset, [train_size + val_size, test_size], generator=generator)
                train_set, _ = random_split(train_val_set, [train_size, val_size], generator=generator)
                train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)

                self.deep_classifier = FaceClassifier(self.current_model, dataset, torch.device("cuda" if torch.cuda.is_available() else "cpu"), threshold=0.6)
                self.deep_classifier.build_centers(train_loader)
            except Exception as e:
                self.status_label.setText(f"Model loaded, but failed to build centers: {e}")

        if 'classifier' in model_parts and 'lda' in model_parts:
            self.current_model = model_parts
            self.model_type = "traditional"
            self.status_label.setText("Loaded traditional model components.")
        else:
            self.status_label.setText("Missing key components (e.g. classifier or lda).")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated, _ = self.process_frame(frame_rgb)
        img = QImage(annotated.data, annotated.shape[1], annotated.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.image_label.setPixmap(pixmap)

    def process_frame(self, image):
        annotated = image.copy()
        prediction_text = ""

        if self.model_type == "deep":
            boxes, _ = self.mtcnn.detect(image)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_img = image[y1:y2, x1:x2]
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            temp_path = tmp.name
                            cv2.imwrite(temp_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

                        classifier = self.deep_classifier if hasattr(self, 'deep_classifier') else None
                        if not classifier:
                            raise ValueError("Classifier not initialized.")

                        name, sim, face_pil = classifier.predict(temp_path)
                        prediction_text = f"{name} ({sim:.2f})" if name else f"Unknown ({sim:.2f})"
                        os.remove(temp_path)
                    except Exception as e:
                        prediction_text = f"Error: {e}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, prediction_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        elif self.model_type == "traditional":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_img, (100, 100)).flatten().reshape(1, -1)
                try:
                    scaler = self.current_model['scaler']
                    pca = self.current_model['pca']
                    lda = self.current_model['lda']
                    clf = self.current_model['classifier']
                    label_dict = self.current_model.get('label_dict', {})

                    face_scaled = scaler.transform(face_resized)
                    face_pca = pca.transform(face_scaled)
                    face_lda = lda.transform(face_pca)
                    prediction = clf.predict(face_lda)[0]
                    prediction_text = label_dict.get(str(prediction), str(prediction))
                except Exception as e:
                    prediction_text = "Error"

                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(annotated, prediction_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return annotated, prediction_text

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())