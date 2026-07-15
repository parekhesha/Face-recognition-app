# Face Recognition App

[![Live Demo](https://img.shields.io/badge/status-live-brightgreen)](https://face-recognition-app-jlxv.onrender.com) [![Made with Flask](https://img.shields.io/badge/backend-Flask-black)](https://flask.palletsprojects.com/) [![OpenCV](https://img.shields.io/badge/CV-OpenCV-blue)](https://opencv.org/)

🔗 **Live demo:** https://face-recognition-app-jlxv.onrender.com


A Flask web app with three features:
1. **Live Detect** – draws a box around any face seen through your webcam.
2. **Register & Recognize** – upload a few photos of a person, then recognize them live via webcam.
3. **Upload & Detect** – upload a photo and get faces detected + labeled.

Uses OpenCV's Haar Cascade for detection and LBPH for recognition — no `dlib`/`cmake`
build required, so it installs cleanly on Windows.

## Setup

1. Copy these files into your `face-recognition-app` folder (alongside your existing `venv`).
2. Activate your virtual environment (PowerShell):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the app:
   ```powershell
   python app.py
   ```
5. Open your browser to **http://127.0.0.1:5000**

## Notes

- Your browser will ask for camera permission — allow it.
- For best recognition results, register 3–5 clear, well-lit photos per person.
- `known_faces/` and `trainer/` are created automatically and store your registered
  faces and trained model — they're excluded from git via `.gitignore`.
- If recognition seems inaccurate, adjust `RECOGNITION_THRESHOLD` in `app.py`
  (lower = stricter matching).
