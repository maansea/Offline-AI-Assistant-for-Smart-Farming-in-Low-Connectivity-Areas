import json
import os
import wave
from pathlib import Path

import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "vosk"
MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", str(MODEL_DIR))


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

    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        return ""

    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        recognizer.AcceptWaveform(data)

    result = recognizer.FinalResult()
    payload = json.loads(result)
    return payload.get("text", "")
