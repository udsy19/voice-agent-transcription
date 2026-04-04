"""Kokoro TTS — high-quality local text-to-speech with 20+ voices.

Uses kokoro-onnx for fast inference. Models auto-download on first use.
Voices are selectable in settings with live preview.
"""

import os
import time
import threading
import subprocess
import logging
import numpy as np
import sounddevice as sd
from logger import get_logger
from config import DATA_DIR

# Suppress noisy phonemizer warnings
logging.getLogger("phonemizer").setLevel(logging.ERROR)

log = get_logger("tts")

MODEL_DIR = str(DATA_DIR / "models")
MODEL_PATH = os.path.join(MODEL_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODEL_DIR, "voices-v1.0.bin")
SAMPLE_RATE = 24000

# All available Kokoro voices with descriptions
VOICES = {
    "af_heart":  {"name": "Heart", "gender": "F", "accent": "American", "desc": "Warm, friendly"},
    "af_alloy":  {"name": "Alloy", "gender": "F", "accent": "American", "desc": "Clear, professional"},
    "af_aoede":  {"name": "Aoede", "gender": "F", "accent": "American", "desc": "Soft, melodic"},
    "af_bella":  {"name": "Bella", "gender": "F", "accent": "American", "desc": "Confident, bright"},
    "af_jessica": {"name": "Jessica", "gender": "F", "accent": "American", "desc": "Casual, natural"},
    "af_kore":   {"name": "Kore", "gender": "F", "accent": "American", "desc": "Gentle, calm"},
    "af_nicole": {"name": "Nicole", "gender": "F", "accent": "American", "desc": "Smooth, articulate"},
    "af_nova":   {"name": "Nova", "gender": "F", "accent": "American", "desc": "Energetic, youthful"},
    "af_river":  {"name": "River", "gender": "F", "accent": "American", "desc": "Relaxed, flowing"},
    "af_sarah":  {"name": "Sarah", "gender": "F", "accent": "American", "desc": "Warm, conversational"},
    "af_sky":    {"name": "Sky", "gender": "F", "accent": "American", "desc": "Light, airy"},
    "am_adam":   {"name": "Adam", "gender": "M", "accent": "American", "desc": "Deep, authoritative"},
    "am_echo":   {"name": "Echo", "gender": "M", "accent": "American", "desc": "Resonant, clear"},
    "am_eric":   {"name": "Eric", "gender": "M", "accent": "American", "desc": "Friendly, approachable"},
    "am_liam":   {"name": "Liam", "gender": "M", "accent": "American", "desc": "Young, dynamic"},
    "am_michael": {"name": "Michael", "gender": "M", "accent": "American", "desc": "Professional, steady"},
    "am_onyx":   {"name": "Onyx", "gender": "M", "accent": "American", "desc": "Rich, powerful"},
    "bf_emma":   {"name": "Emma", "gender": "F", "accent": "British", "desc": "Elegant, refined"},
    "bf_isabella": {"name": "Isabella", "gender": "F", "accent": "British", "desc": "Graceful, poised"},
    "bm_daniel": {"name": "Daniel", "gender": "M", "accent": "British", "desc": "Classic, distinguished"},
    "bm_george": {"name": "George", "gender": "M", "accent": "British", "desc": "Warm, trustworthy"},
    "bm_lewis":  {"name": "Lewis", "gender": "M", "accent": "British", "desc": "Modern, articulate"},
}

PREVIEW_TEXT = "Hi, I'm your voice assistant. How can I help you today?"

_kokoro = None
_kokoro_lock = threading.Lock()
_playing = False


def _ensure_models():
    """Download models if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    base_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    for fname in ["kokoro-v1.0.onnx", "voices-v1.0.bin"]:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            log.info("Downloading %s...", fname)
            try:
                import urllib.request
                import ssl
                try:
                    import certifi
                    ctx = ssl.create_default_context(cafile=certifi.where())
                except ImportError:
                    ctx = ssl.create_default_context()
                urllib.request.urlretrieve(f"{base_url}/{fname}", path,
                                          context=ctx if hasattr(urllib.request, 'urlretrieve') else None)
                # Actually urlretrieve doesn't take context, use urlopen
                req = urllib.request.Request(f"{base_url}/{fname}")
                with urllib.request.urlopen(req, context=ctx) as resp, open(path, 'wb') as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                log.info("Downloaded %s", fname)
            except Exception as e:
                log.error("Failed to download %s: %s", fname, e)
                return False
    return True


def _get_kokoro():
    """Lazy-load Kokoro TTS engine."""
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    with _kokoro_lock:
        if _kokoro is not None:
            return _kokoro
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
            if not _ensure_models():
                return None
        try:
            from kokoro_onnx import Kokoro
            _kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
            log.info("Kokoro TTS loaded")
            return _kokoro
        except Exception as e:
            log.error("Failed to load Kokoro: %s", e)
            return None


def speak(text: str, voice: str = "af_heart"):
    """Speak text using Kokoro TTS. Non-blocking."""
    global _playing
    if _playing:
        stop()

    def run():
        global _playing
        _playing = True
        try:
            kokoro = _get_kokoro()
            if kokoro is None:
                # Fallback to macOS say
                _fallback_speak(text)
                return
            audio, sr = kokoro.create(text, voice=voice, speed=1.0)
            if audio is not None and len(audio) > 0:
                sd.play(audio, samplerate=sr)
                sd.wait()
        except Exception as e:
            log.error("TTS failed: %s, falling back to macOS say", e)
            _fallback_speak(text)
        finally:
            _playing = False

    threading.Thread(target=run, daemon=True).start()


def speak_sync(text: str, voice: str = "af_heart") -> np.ndarray | None:
    """Speak text and return audio array (for preview). Blocking."""
    try:
        kokoro = _get_kokoro()
        if kokoro is None:
            return None
        audio, sr = kokoro.create(text, voice=voice, speed=1.0)
        if audio is not None and len(audio) > 0:
            sd.play(audio, samplerate=sr)
            sd.wait()
            return audio
    except Exception as e:
        log.error("TTS preview failed: %s", e)
    return None


def stop():
    """Stop any currently playing audio."""
    global _playing
    try:
        sd.stop()
    except Exception:
        pass
    _playing = False


def _fallback_speak(text: str):
    """Fallback to macOS say command."""
    try:
        subprocess.Popen(
            ["say", "-v", "Samantha", "-r", "190", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def get_voices() -> dict:
    """Return all available voices with metadata."""
    return VOICES


def is_available() -> bool:
    """Check if Kokoro models are downloaded."""
    return os.path.exists(MODEL_PATH) and os.path.exists(VOICES_PATH)
