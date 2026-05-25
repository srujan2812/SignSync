import numpy as np

def normalize_landmarks(landmarks_list):
    """
    Normalizes hand landmarks to make the model invariant to hand position and scale.
    1. Translates landmarks so the wrist (index 0) is at (0, 0, 0).
    2. Scales landmarks so the maximum distance from the wrist is 1.
    """
    if not landmarks_list or len(landmarks_list) != 63:
        return landmarks_list

    # Reshape to (21, 3) for easier processing
    temp_landmarks = np.array(landmarks_list).reshape(-1, 3)
    
    # 1. Coordinate shift: wrist (index 0) becomes (0,0,0)
    base_x, base_y, base_z = temp_landmarks[0]
    temp_landmarks[:, 0] -= base_x
    temp_landmarks[:, 1] -= base_y
    temp_landmarks[:, 2] -= base_z
        
    # 2. Scaling: normalize by max absolute value to keep values between -1 and 1
    max_val = np.max(np.abs(temp_landmarks))
    if max_val != 0:
        temp_landmarks = temp_landmarks / max_val
        
    return temp_landmarks.flatten().tolist()
