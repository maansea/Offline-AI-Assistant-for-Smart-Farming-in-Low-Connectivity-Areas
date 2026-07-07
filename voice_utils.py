import json
import os
import wave
from pathlib import Path

import speech_recognition as sr
from vosk import KaldiRecognizer, Model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "vosk"
MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", str(MODEL_DIR))


def build_transcription_error_message(audio_path: str = "", model_path: str = MODEL_PATH) -> str:
    if not audio_path or not os.path.exists(audio_path):
        return "The uploaded audio file could not be found."

    if not os.path.isdir(model_path) or not os.listdir(model_path):
        return "Offline speech model is not available. Please place a Vosk model in models/vosk before using voice transcription."

    return "Audio was uploaded successfully but could not be transcribed. Please try a clearer recording or upload a WAV file."


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH):
        return MODEL_PATH

    os.makedirs(MODEL_PATH, exist_ok=True)
    return MODEL_PATH


def transcribe_audio_file(audio_path: str, sample_rate: int = 16000) -> str:
    ensure_model_downloaded()

    if not os.path.exists(audio_path):
        return ""

    try:
        model = Model(MODEL_PATH)
    except Exception:
        return ""

    try:
        wf = wave.open(audio_path, "rb")
    except Exception:
        return ""

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        return ""

    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        try:
            recognizer.AcceptWaveform(data)
        except Exception:
            continue

    try:
        result = recognizer.FinalResult()
        payload = json.loads(result)
        return payload.get("text", "")
    except Exception:
        return ""


def transcribe_microphone_stream(duration: int = 5) -> str:
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
    except Exception:
        return "Microphone access failed. Please allow microphone permission and try again."

    try:
        return recognizer.recognize_sphinx(audio)
    except sr.UnknownValueError:
        return "Speech not recognized. Please try again with clearer speech."
    except sr.RequestError:
        return "Offline speech recognition is unavailable in this environment."
