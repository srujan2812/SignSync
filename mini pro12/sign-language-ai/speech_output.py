import pyttsx3 
import threading

engine = pyttsx3.init() 

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 150)

def _speak_thread(text):
    try:
        engine.say(text) 
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")

def speak(text): 
    if not text:
        return
    print(f"Speaking: {text}")
    threading.Thread(target=_speak_thread, args=(text,), daemon=True).start()