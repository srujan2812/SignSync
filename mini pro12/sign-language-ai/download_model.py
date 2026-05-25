import urllib.request
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = os.path.join(model_dir, "hand_landmarker.task")

if not os.path.exists(model_path):
    print(f"Downloading model to {model_path}...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")
else:
    print("Model already exists.")