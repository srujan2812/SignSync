import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import sys
import os

# Add sign-language-ai to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "sign-language-ai"))
from utils import normalize_landmarks

app = Flask(__name__)
app.config['SECRET_KEY'] = 'signsync_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
# Note: Adjusting path to point into sign-language-ai/models
model_path = os.path.join(script_dir, "sign-language-ai", "models", "sign_model.pkl")

gesture_model = None
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        gesture_model = pickle.load(f)
    if isinstance(gesture_model, dict) and "model" in gesture_model:
        gesture_model = gesture_model["model"]
else:
    print(f"Warning: Model not found at {model_path}")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('predict')
def handle_prediction(data):
    if gesture_model is None:
        emit('prediction_result', {'error': 'Model not loaded'})
        return

    try:
        landmarks = data['landmarks'] # Expected list of 63 values (x,y,z * 21)
        normalized_landmarks = normalize_landmarks(landmarks)
        features = np.array(normalized_landmarks).reshape(1, -1)
        
        # Get prediction
        prediction = gesture_model.predict(features)[0]
        
        # Get confidence (if model supports it)
        confidence = 0
        if hasattr(gesture_model, "predict_proba"):
            probabilities = gesture_model.predict_proba(features)
            confidence = float(np.max(probabilities) * 100)

        # Filter out low-confidence predictions to prevent false positives
        if confidence < 65: # Require at least 65% certainty
            emit('prediction_result', {
                'gesture': 'NONE',
                'confidence': round(confidence, 2),
                'status': 'Uncertain'
            })
            return

        emit('prediction_result', {
            'gesture': prediction.upper(),
            'confidence': round(confidence, 2),
            'status': 'Static'
        })
    except Exception as e:
        emit('prediction_result', {'error': str(e)})

if __name__ == '__main__':
    # host='0.0.0.0' allows other devices on your Wi-Fi to access the app
    socketio.run(app, host='0.0.0.0', debug=True, port=5000)
