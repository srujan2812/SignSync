import os
import numpy as np
from utils import normalize_landmarks

def migrate_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset")
    
    if not os.path.exists(dataset_path):
        print("No dataset found to normalize.")
        return

    count = 0
    for gesture in os.listdir(dataset_path):
        gesture_path = os.path.join(dataset_path, gesture)
        if not os.path.isdir(gesture_path):
            continue
            
        print(f"Normalizing gesture: {gesture}")
        for file in os.listdir(gesture_path):
            if file.endswith(".npy"):
                file_path = os.path.join(gesture_path, file)
                data = np.load(file_path).tolist()
                normalized_data = normalize_landmarks(data)
                np.save(file_path, np.array(normalized_data))
                count += 1
                
    print(f"Finished! Normalized {count} samples.")

if __name__ == "__main__":
    migrate_dataset()
