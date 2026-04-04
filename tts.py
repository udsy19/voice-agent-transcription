"""TTS — Kokoro ONNX (local, fast) → macOS say (fallback).

No cloud dependency. All voices run locally via ONNX inference.
"""

import os
import threading
import subprocess
import logging
import numpy as np
import sounddevice as sd
from logger import get_logger
from config import DATA_DIR

log = get_logger("tts")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DIR = str(DATA_DIR / "models")
KOKORO_MODEL = os.path.join(MODEL_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(MODEL_DIR, "voices-v1.0.bin")

VOICES = {
    "af_heart":   {"name": "Heart", "gender": "F", "accent": "US", "desc": "Warm, friendly"},
    "af_alloy":   {"name": "Alloy", "gender": "F", "accent": "US", "desc": "Clear, professional"},
    "af_bella":   {"name": "Bella", "gender": "F", "accent": "US", "desc": "Confident, bright"},
    "af_nicole":  {"name": "Nicole", "gender": "F", "accent": "US", "desc": "Smooth, articulate"},
    "af_nova":    {"name": "Nova", "gender": "F", "accent": "US", "desc": "Energetic, youthful"},
    "af_sky":     {"name": "Sky", "gender": "F", "accent": "US", "desc": "Light, airy"},
    "am_adam":    {"name": "Adam", "gender": "M", "accent": "US", "desc": "Deep, authoritative"},
    "am_eric":    {"name": "Eric", "gender": "M", "accent": "US", "desc": "Friendly, approachable"},
    "am_michael": {"name": "Michael", "gender": "M", "accent": "US", "desc": "Professional, steady"},
    "bf_emma":    {"name": "Emma", "gender": "F", "accent": "UK", "desc": "Elegant, refined"},
    "bm_daniel":  {"name": "Daniel", "gender": "M", "accent": "UK", "desc": "Classic, distinguished"},
    "bm_george":  {"name": "George", "gender": "M", "accent": "UK", "desc": "Warm, trustworthy"},
}

PREVIEW_TEXT = "Hi, I'm your voice assistant. How can I help you today?"

_kokoro = None
_kokoro_lock = threading.Lock()
_playing = False


def _get_kokoro():
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    with _kokoro_lock:
        if _kokoro is not None:
            return _kokoro
        if not os.path.exists(KOKORO_MODEL) or not os.path.exists(KOKORO_VOICES):
            log.warning("Kokoro models not found at %s", MODEL_DIR)
            return None
        try:
            from kokoro_onnx import Kokoro
            _kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
            log.info("Kokoro TTS loaded")
            return _kokoro
        except Exception as e:
            log.warning("Kokoro unavailable: %s", str(e)[:60])
            return None


def speak(text: str, voice: str = "af_heart"):
    """Speak text. Preprocesses for natural speech, then Kokoro → macOS say."""
    global _playing
    if _playing:
        stop()

    # Preprocess text for natural speech
    from speech_prep import prepare_for_speech
    spoken_text = prepare_for_speech(text)

    def run():
        global _playing
        _playing = True
        try:
            kokoro = _get_kokoro()
            if kokoro:
                audio, sr = kokoro.create(spoken_text, voice=voice, speed=1.05)
                if audio is not None and len(audio) > 0:
                    try:
                        sd.play(audio, samplerate=sr)
                        sd.wait()
                    except Exception:
                        try:
                            sd.play(audio, samplerate=sr, device=sd.default.device[1])
                            sd.wait()
                        except Exception:
                            _speak_macos(text)
                    return
            _speak_macos(spoken_text)
        except Exception as e:
            log.warning("TTS failed: %s", e)
            _speak_macos(spoken_text)
        finally:
            _playing = False

    threading.Thread(target=run, daemon=True).start()


def speak_sync(text: str, voice: str = "af_heart"):
    """Speak text synchronously (for preview)."""
    kokoro = _get_kokoro()
    if kokoro:
        try:
            audio, sr = kokoro.create(text, voice=voice, speed=1.05)
            if audio is not None and len(audio) > 0:
                sd.play(audio, samplerate=sr)
                sd.wait()
                return
        except Exception as e:
            log.warning("Preview failed: %s", e)
    _speak_macos(text)


def stop():
    global _playing
    try:
        sd.stop()
    except Exception:
        pass
    _playing = False


def _speak_macos(text: str):
    try:
        subprocess.Popen(["say", "-v", "Samantha", "-r", "190", text],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def get_voices() -> dict:
    return VOICES


def is_available() -> bool:
    return os.path.exists(KOKORO_MODEL) and os.path.exists(KOKORO_VOICES)
