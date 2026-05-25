import os 
import numpy as np 
import pickle 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "dataset") 
X = [] 
y = [] 

if not os.path.exists(dataset_path):
    print(f"Error: '{dataset_path}' folder not found. Please collect data first.")
    exit()

for gesture in os.listdir(dataset_path): 
    gesture_path = os.path.join(dataset_path, gesture) 
    
    if not os.path.isdir(gesture_path):
        continue

    print(f"Loading samples for gesture: {gesture}")
    files = os.listdir(gesture_path)
    for file in files: 
        if file.endswith('.npy'):
            file_path = os.path.join(gesture_path, file) 
            data = np.load(file_path) 
            X.append(data) 
            y.append(gesture) 

if len(X) == 0:
    print("Error: No data found in dataset folders. Please collect data using collect_data.py.")
    exit()

X = np.array(X) 
y = np.array(y) 

print("Dataset shape:", X.shape) 

X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42, stratify=y
) 

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100) 
model.fit(X_train, y_train) 

predictions = model.predict(X_test) 
accuracy = accuracy_score(y_test, predictions) 

print("-" * 30)
print(f"Model Accuracy: {accuracy * 100:.2f}%") 
print("-" * 30)

model_dir = os.path.join(script_dir, "models")
os.makedirs(model_dir, exist_ok=True) 
model_file = os.path.join(model_dir, "sign_model.pkl")

with open(model_file, "wb") as f:
    pickle.dump(model, f) 

print(f"Model saved successfully to {model_file}")