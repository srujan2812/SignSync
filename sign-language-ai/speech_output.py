import pyttsx3 
import threading

# Initialize the speech engine
engine = pyttsx3.init() 

# Optional: Configure voice properties
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) # Usually 0 is male, 1 is female
engine.setProperty('rate', 150)           # Speed of speech

def _speak_thread(text):
    """Internal function to be run in a separate thread."""
    try:
        engine.say(text) 
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")

def speak(text): 
    """Converts text to speech in a non-blocking thread."""
    if not text:
        return
    print(f"Speaking: {text}")
    threading.Thread(target=_speak_thread, args=(text,), daemon=True).start()