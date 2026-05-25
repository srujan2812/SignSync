# SignSync - AI Sign Language Recognition

SignSync is a real-time American Sign Language (ASL) recognition system... built with Python, MediaPipe, and Flask. It uses a Random Forest classifier to interpret hand gestures from a webcam feed and provides a modern web interface for interaction.

## Features
- **Real-time Recognition**: Fast and accurate ASL fingerspelling interpretation.
- **Web Interface**: Modern, responsive dashboard with light/dark mode.
- **Hand Landmark Normalization**: Position and scale-invariant recognition for high accuracy.
- **Audio Output**: Text-to-speech for interpreted signs.

## Tech Stack
- **Frontend**: HTML5, CSS3 (Custom variables, responsive design), JavaScript, MediaPipe Hands JS.
- **Backend**: Flask, Flask-SocketIO (Real-time communication).
- **AI/ML**: MediaPipe (Hand tracking), Scikit-Learn (Random Forest), NumPy.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`.
3. Activate the environment: `.\venv\Scripts\activate` (Windows).
4. Install dependencies: `pip install -r requirements.txt`.
5. Run the app: `python flask_app.py`.
6. Open `http://localhost:5000` in your browser.

## Project Structure
- `flask_app.py`: Main entry point for the web application.
- `sign-language-ai/`: Core AI logic and models.
- `static/` & `templates/`: Frontend assets and HTML.
- `requirements.txt`: Project dependencies.
