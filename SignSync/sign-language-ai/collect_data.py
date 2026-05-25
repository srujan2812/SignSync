import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

gesture = input("Enter the gesture name (e.g., A, B, C, SPACE): ").strip().upper()
script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "dataset", gesture)
os.makedirs(folder, exist_ok=True)

model_path = os.path.join(script_dir, "models", "hand_landmarker.task")
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}. Please run download_model.py first.")
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

cap = cv2.VideoCapture(0)
sample_count = 0

print(f"Starting data collection for: {gesture}")
print("Show your hand and perform the gesture. Press ESC to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            landmarks_list = []
            
            for landmark in hand_landmarks:
                landmarks_list.append(landmark.x)
                landmarks_list.append(landmark.y)
                landmarks_list.append(landmark.z)
                
                px = int(landmark.x * frame.shape[1])
                py = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            for start_idx, end_idx in connections:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_p = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_p = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, start_p, end_p, (255, 0, 0), 2)

            file_path = f"{folder}/sample_{sample_count}.npy"
            np.save(file_path, landmarks_list)
            
            sample_count += 1
            if sample_count % 10 == 0:
                print(f"Samples collected: {sample_count}")

    cv2.putText(frame, f"Gesture: {gesture} | Samples: {sample_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Collecting A-Z Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

print(f"Collection finished. Total samples for '{gesture}': {sample_count}")
cap.release()
cv2.destroyAllWindows()
try:
    detector.close()
    detector = None
except Exception:
    pass