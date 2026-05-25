import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
import time
from collections import deque, Counter
from speech_output import speak
from utils import normalize_landmarks

# 1. Load the trained AI model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, "models", "sign_model.pkl")

if not os.path.exists(model_file):
    print(f"Error: Trained model not found at {model_file}. Please run train_model.py first.")
    exit()

with open(model_file, "rb") as f:
    gesture_model = pickle.load(f)

# If the model is a dictionary, extract the classifier
if isinstance(gesture_model, dict):
    if "model" in gesture_model:
        gesture_model = gesture_model["model"]
    else:
        print("Error: The model file is in an unrecognized dictionary format.")
        exit()

# Verify the model's expected features
if hasattr(gesture_model, "n_features_in_"):
    expected_features = gesture_model.n_features_in_
    if expected_features != 63:
        print(f"Error: Model expects {expected_features} features, but we are providing 63.")
        print("Please retrain the model using train_model.py.")
        exit()

# 2. Initialize MediaPipe Hand Landmarker (Tasks API)
model_path = os.path.join(script_dir, "models", "hand_landmarker.task")
if not os.path.exists(model_path):
    print(f"Error: MediaPipe model not found at {model_path}. Please run download_model.py first.")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Finger Spelling Configuration
prediction_buffer = deque(maxlen=10) # Majority voting over last 10 frames
current_word = ""                    # Joined letters
last_added_time = 0                  # Time when last letter was added
ADD_DELAY = 2                    # Seconds to wait between letters
STABILITY_THRESHOLD = 7              # Most common letter must appear 7/10 times
last_hand_time = time.time()
RESET_TIMEOUT = 3.0                  # Seconds without a hand to clear the word

cap = cv2.VideoCapture(0)

print("SignSync - Robust ASL Fingerspelling System Started. Press ESC to exit.")
print("Press 'c' to clear the current word.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    current_time = time.time()

    # 4. Process Hand Landmarks
    if detection_result.hand_landmarks:
        last_hand_time = current_time
        for hand_landmarks in detection_result.hand_landmarks:
            landmarks_list = []
            for landmark in hand_landmarks:
                landmarks_list.append(landmark.x)
                landmarks_list.append(landmark.y)
                landmarks_list.append(landmark.z)
                
                # Visual landmarks
                px = int(landmark.x * frame.shape[1])
                py = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            # Predict current letter and add to buffer
            normalized_features = normalize_landmarks(landmarks_list)
            features = np.array(normalized_features).reshape(1, -1)
            prediction = gesture_model.predict(features)
            predicted_letter = prediction[0].upper()
            prediction_buffer.append(predicted_letter)

            # 5. Stability & Word Formation Logic (Majority Voting)
            if len(prediction_buffer) == 10:
                # Find most common prediction in the buffer
                counts = Counter(prediction_buffer)
                most_common, count = counts.most_common(1)[0]
                
                # Only accept if most common letter is stable (7/10) and enough time has passed
                if count >= STABILITY_THRESHOLD:
                    if current_time - last_added_time > ADD_DELAY:
                        if most_common == "SPACE":
                            if current_word and not current_word.endswith(" "):
                                # Safely get the last word
                                words = current_word.strip().split()
                                if words:
                                    speak(words[-1]) # Speak the last word
                                current_word += " "
                                last_added_time = current_time
                        else:
                            # Avoid duplicate letters unless explicitly held/re-signed
                            # If we want to allow doubles, the 1.5s delay handles it, 
                            # but we clear the buffer to make it intentional.
                            current_word += most_common
                            last_added_time = current_time
                            prediction_buffer.clear() # Require 10 fresh stable frames for next char
                            print(f"Added Letter: {most_common}")

            # Display current live prediction and stability info
            cv2.putText(frame, f"Live: {predicted_letter}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Progress bar for stability (based on count of most common)
            if len(prediction_buffer) == 10:
                # Using the counts calculated above (most_common, count)
                bar_width = int((count / 10) * 200)
                cv2.rectangle(frame, (50, 65), (50 + bar_width, 75), (0, 255, 0), -1)
                cv2.putText(frame, f"Stable: {most_common}", (260, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Show cooldown progress
                time_passed = current_time - last_added_time
                if time_passed < ADD_DELAY:
                    cooldown_pct = time_passed / ADD_DELAY
                    cv2.rectangle(frame, (50, 85), (50 + int(cooldown_pct * 200), 90), (255, 255, 0), -1)

    else:
        # If no hand is seen for RESET_TIMEOUT seconds, speak and clear the word
        if current_time - last_hand_time > RESET_TIMEOUT and current_word:
            if current_word.strip():
                speak(current_word.strip())
            current_word = ""
            prediction_buffer.clear()
            print("Word cleared due to inactivity.")

    # 6. Display Formed Word
    cv2.putText(frame, f"Word: {current_word}", (50, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    # 7. UI Info
    cv2.putText(frame, "ESC: Exit | 'c': Clear", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("SignSync - AI Sign Language Interpreter", frame)

    # 8. Handle Keyboard Inputs
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('c'): # Clear word manually
        current_word = ""
        prediction_buffer.clear()
        print("Word cleared manually.")

# 9. Release resources
cap.release()
cv2.destroyAllWindows()
try:
    detector.close()
    detector = None
except Exception:
    pass