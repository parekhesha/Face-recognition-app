import os
import io
import base64
import pickle

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_DIR, "labels.pickle")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

FACE_SIZE = (200, 200)
# LBPH "confidence" is actually a distance -- lower is a better match.
# Anything above this is treated as "Unknown".
RECOGNITION_THRESHOLD = 75

face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")
)
if face_cascade.empty():
    raise RuntimeError(
        "Failed to load haarcascade_frontalface_default.xml from models/ folder"
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()

label_map = {}  # {numeric_id: name}


def load_model():
    """Load the trained recognizer + labels from disk, if they exist."""
    global label_map
    if os.path.exists(TRAINER_FILE) and os.path.exists(LABELS_FILE):
        recognizer.read(TRAINER_FILE)
        with open(LABELS_FILE, "rb") as f:
            label_map = pickle.load(f)
    else:
        label_map = {}


def train_model():
    """Retrain the LBPH recognizer from everything in known_faces/."""
    global label_map
    faces = []
    labels = []
    label_map = {}
    current_id = 0
    name_to_id = {}

    for name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        if name not in name_to_id:
            name_to_id[name] = current_id
            label_map[current_id] = name
            current_id += 1

        person_id = name_to_id[name]

        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            faces.append(img)
            labels.append(person_id)

    if len(faces) == 0:
        return 0, 0

    recognizer.train(faces, np.array(labels))
    recognizer.write(TRAINER_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    return len(name_to_id), len(faces)


def decode_base64_image(data_url):
    """Turn a 'data:image/jpeg;base64,...' string into a cv2 BGR image."""
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(binary)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def encode_image_to_base64(cv2_img):
    """Turn a cv2 BGR image into a 'data:image/jpeg;base64,...' string."""
    _, buffer = cv2.imencode(".jpg", cv2_img)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def detect_faces(gray_img):
    return face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )


@app.route("/")
def index():
    people = sorted(
        [
            name
            for name in os.listdir(KNOWN_FACES_DIR)
            if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name))
        ]
    )
    return render_template("index.html", people=people)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Tab 1: plain face detection, no recognition."""
    data = request.get_json()
    img = decode_base64_image(data["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return jsonify({"image": encode_image_to_base64(img), "count": len(faces)})


@app.route("/api/register", methods=["POST"])
def api_register():
    """Tab 2a: save labeled face images for a person, then retrain."""
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    images = request.files.getlist("images")
    if not images:
        return jsonify({"error": "At least one image is required"}), 400

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing = len(os.listdir(person_dir))
    saved = 0

    for i, file_storage in enumerate(images):
        pil_img = Image.open(file_storage.stream).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        if len(faces) == 0:
            continue

        # Use the largest detected face in the image
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
        out_path = os.path.join(person_dir, f"{existing + saved}.jpg")
        cv2.imwrite(out_path, face_crop)
        saved += 1

    if saved == 0:
        return jsonify({"error": "No face detected in the uploaded image(s)"}), 400

    people_count, sample_count = train_model()

    return jsonify(
        {
            "status": "ok",
            "name": name,
            "saved": saved,
            "people_count": people_count,
            "sample_count": sample_count,
        }
    )


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Tab 2b: detect + recognize known faces from a webcam frame."""
    data = request.get_json()
    img = decode_base64_image(data["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    results = []
    has_model = os.path.exists(TRAINER_FILE) and len(label_map) > 0

    for (x, y, w, h) in faces:
        name = "Unknown"
        confidence = None

        if has_model:
            face_crop = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
            label_id, distance = recognizer.predict(face_crop)
            if distance <= RECOGNITION_THRESHOLD and label_id in label_map:
                name = label_map[label_id]
                confidence = round(100 - min(distance, 100), 1)

        color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_text = name if confidence is None else f"{name} ({confidence}%)"
        cv2.putText(
            img, label_text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        results.append({"name": name, "confidence": confidence})

    return jsonify({"image": encode_image_to_base64(img), "results": results})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Tab 3: detect + recognize faces in an uploaded static image."""
    file_storage = request.files.get("image")
    if file_storage is None:
        return jsonify({"error": "No image uploaded"}), 400

    pil_img = Image.open(file_storage.stream).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    results = []
    has_model = os.path.exists(TRAINER_FILE) and len(label_map) > 0

    for (x, y, w, h) in faces:
        name = "Unknown"
        confidence = None

        if has_model:
            face_crop = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
            label_id, distance = recognizer.predict(face_crop)
            if distance <= RECOGNITION_THRESHOLD and label_id in label_map:
                name = label_map[label_id]
                confidence = round(100 - min(distance, 100), 1)

        color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_text = name if confidence is None else f"{name} ({confidence}%)"
        cv2.putText(
            img, label_text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        results.append({"name": name, "confidence": confidence})

    return jsonify(
        {"image": encode_image_to_base64(img), "count": len(faces), "results": results}
    )


load_model()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
