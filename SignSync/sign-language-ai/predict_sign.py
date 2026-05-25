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

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, "models", "sign_model.pkl")
if not os.path.exists(model_file):
    print(f"Error: Trained model not found at {model_file}. Please run train_model.py first.")
    exit()

with open(model_file, "rb") as f:
    gesture_model = pickle.load(f)

if isinstance(gesture_model, dict):
    if "model" in gesture_model:
        gesture_model = gesture_model["model"]
    else:
        print("Error: The model file is in an unrecognized dictionary format.")
        exit()

if hasattr(gesture_model, "n_features_in_"):
    expected_features = gesture_model.n_features_in_
    if expected_features != 63:
        print(f"Error: Model expects {expected_features} features, but we are providing 63.")
        print("Please retrain the model using train_model.py.")
        exit()

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

prediction_buffer = deque(maxlen=10) 
current_word = ""                   
last_added_time = 0           
ADD_DELAY = 2                 
STABILITY_THRESHOLD = 7            
last_hand_time = time.time()
RESET_TIMEOUT = 5.0                

cap = cv2.VideoCapture(0)


window_name = "SignSync Interactive"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


print(f"{window_name} System Started. Press ESC to exit.")
print("Press 'c' to clear the current word.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        if win_w > 0 and win_h > 0:
            frame = cv2.resize(frame, (win_w, win_h))
    except:
        pass

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    current_time = time.time()


    if detection_result.hand_landmarks:
        last_hand_time = current_time
        for hand_landmarks in detection_result.hand_landmarks:
            landmarks_list = []
            for landmark in hand_landmarks:
                landmarks_list.append(landmark.x)
                landmarks_list.append(landmark.y)
                landmarks_list.append(landmark.z)
                
                
                px = int(landmark.x * frame.shape[1])
                py = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            
            features = np.array(landmarks_list).reshape(1, -1)
            prediction = gesture_model.predict(features)
            predicted_letter = prediction[0].upper()
            prediction_buffer.append(predicted_letter)

            
            if len(prediction_buffer) == 10:
               
                counts = Counter(prediction_buffer)
                most_common, count = counts.most_common(1)[0]
                
                
                if count >= STABILITY_THRESHOLD:
                    if current_time - last_added_time > ADD_DELAY:
                        if most_common == "SPACE":
                            if current_word and not current_word.endswith(" "):
                               
                                words = current_word.strip().split()
                                if words:
                                    speak(words[-1]) 
                                current_word += " "
                                last_added_time = current_time
                        else:
                            current_word += most_common
                            speak(most_common) 
                            last_added_time = current_time
                            prediction_buffer.clear() 
                            print(f"Added Letter: {most_common}")

          
            cv2.putText(frame, f"LIVE: {predicted_letter}", (30, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
            
            
            if len(prediction_buffer) == 10:
                bar_x, bar_y = 30, 110
                bar_w, bar_h = 400, 20
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1) # Background
                fill_w = int((count / 10) * bar_w)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1) # Fill
                cv2.putText(frame, f"STABLE: {most_common}", (bar_x + bar_w + 15, bar_y + 16), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
            
                time_passed = current_time - last_added_time
                if time_passed < ADD_DELAY:
                    cd_w = int((time_passed / ADD_DELAY) * bar_w)
                    cv2.rectangle(frame, (bar_x, bar_y + 30), (bar_x + cd_w, bar_y + 35), (255, 255, 0), -1)

    else:
        if current_time - last_hand_time > RESET_TIMEOUT and current_word:
            if current_word.strip():
                speak(current_word.strip())
            current_word = ""
            prediction_buffer.clear()
            print("Word cleared due to inactivity.")


    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame.shape[0] - 150), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)


    cv2.putText(frame, f"WORD: {current_word}", (50, frame.shape[0] - 60), 
                cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 4)
    

    cv2.putText(frame, "ESC: Exit | 'C': Clear", (frame.shape[1] - 300, frame.shape[0] - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('c'):
        current_word = ""
        prediction_buffer.clear()
        print("Word cleared manually.")


cap.release()
cv2.destroyAllWindows()
try:
    detector.close()
    detector = None
except Exception:
    pass