import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import random
from collections import Counter
from flask import Flask, render_template_string, request, redirect, send_file
import webbrowser
import threading
import time

# Flask app setup
app = Flask(__name__)

DATASET_DIR = "filter_enhanced"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (100, 100)

# Model training function
def train_model():
    X, y, label_dict = [], [], {}
    label_id = 0
    print("[INFO] Loading dataset from:", DATASET_DIR)
    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_dict[label_id] = person_name
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.equalizeHist(cv2.resize(img, FACE_SIZE))
            # Original
            X.append(img.flatten())
            y.append(label_id)
            # Flip
            flipped = cv2.flip(img, 1)
            X.append(flipped.flatten())
            y.append(label_id)
            # Noise
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            noisy = cv2.add(img, noise)
            X.append(noisy.flatten())
            y.append(label_id)
        label_id += 1

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.98, svd_solver='full')
    X_pca = pca.fit_transform(X_scaled)
    lda = LDA(n_components=min(len(np.unique(y)) - 1, 8))
    X_lda = lda.fit_transform(X_pca, y)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_lda, y)
    with open("face_model.pkl", "wb") as f:
        pickle.dump((scaler, pca, lda, clf, label_dict), f)
    print("[INFO] Model trained and saved as 'face_model.pkl'.")

# Train on launch if model file does not exist
if not os.path.exists("face_model.pkl"):
    train_model()

# Load trained model
with open("face_model.pkl", "rb") as f:
    scaler, pca, lda, clf, label_dict = pickle.load(f)

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Background capture thread
class CameraHandler:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.frame = None
        self.label = "Unknown"
        self.confidence = 0.0
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            label = "Unknown"
            confidence = 0.0
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                face = cv2.equalizeHist(cv2.resize(roi_gray, FACE_SIZE)).flatten().reshape(1, -1)
                face = scaler.transform(face)
                face = pca.transform(face)
                face = lda.transform(face)
                probs = clf.predict_proba(face)[0]
                pred = np.argmax(probs)
                confidence = probs[pred]
                if confidence > 0.55:
                    label = label_dict[pred]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', frame)
            with open("last_frame.jpg", "wb") as f:
                f.write(jpeg.tobytes())
            self.label = label
            self.confidence = confidence

    def stop(self):
        self.running = False
        self.camera.release()

camera_handler = CameraHandler()
def capture_frame():
    return camera_handler.label, camera_handler.confidence

@app.route('/')
def live():
    label, confidence = capture_frame()
    unknown_html = """
    <form method='POST' action='/register'>
        <input type='text' name='name' placeholder='Enter your name'>
        <input type='submit' value='Register'>
    </form>
    """ if label == "Unknown" else ""
    return render_template_string(f'''
        <h2>Live Recognition</h2>
        <img id="live" src="/frame?0" width="500"><br>
        <p>Prediction: <b>{label}</b></p>
        {unknown_html}
        <script>
        setInterval(() => {{
            const img = document.getElementById("live");
            img.src = "/frame?" + new Date().getTime();
        }}, 200);
        </script>
    ''')

@app.route('/frame')
def frame():
    return send_file("last_frame.jpg", mimetype='image/jpeg')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    cap = cv2.VideoCapture(0)
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    count = 0
    max_images = 15
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, FACE_SIZE)
            img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(img_path, roi)
            count += 1
        time.sleep(0.1)
    cap.release()
    train_model()
        # 训练后显示结果图
    with open("face_model.pkl", "rb") as f:
        scaler, pca, lda, clf, label_dict = pickle.load(f)

    # 小样本评估
    X_val, y_val = [], []
    label_map = {}
    label_id = 0
    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_map[label_id] = person_name
        for img_name in os.listdir(person_dir)[:5]:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.equalizeHist(cv2.resize(img, FACE_SIZE)).flatten()
            X_val.append(img)
            y_val.append(label_id)
        label_id += 1

    X_val = scaler.transform(X_val)
    X_val = pca.transform(X_val)
    X_val = lda.transform(X_val)
    y_pred = clf.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_map[i] for i in sorted(label_map)])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Model Confusion Matrix")
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/confusion_matrix.png")

    return render_template_string("""
        <h3 style='color: green;'>✅ Registration successful!</h3>
        <p>Model retrained. Here is the updated performance:</p>
        <img src='/static/confusion_matrix.png' width='600'>
        <br><a href='/'>Back to Recognition</a>
    """)


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.25, open_browser).start()
    app.run(debug=True)
