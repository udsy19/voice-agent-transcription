"""TTS — ElevenLabs (online, premium) → Kokoro (offline fallback) → macOS say.

Priority:
1. ElevenLabs if API key configured + internet available
2. Kokoro ONNX if models downloaded (local, no internet needed)
3. macOS `say` command (always available)
"""

import os
import io
import time
import threading
import subprocess
import logging
import numpy as np
import sounddevice as sd
from logger import get_logger
from config import DATA_DIR

log = get_logger("tts")

# Suppress noisy warnings
logging.getLogger("phonemizer").setLevel(logging.ERROR)

# ── ElevenLabs ──────────────────────────────────────────────────────────────

try:
    from config import ELEVENLABS_API_KEY
    ELEVENLABS_KEY = ELEVENLABS_API_KEY
except ImportError:
    ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Default ElevenLabs voices (subset)
ELEVENLABS_VOICES = {
    "sarah":    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "desc": "Mature, confident"},
    "roger":    {"id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Roger", "desc": "Laid-back, casual"},
    "laura":    {"id": "FGY2WhTYpPnrIDTdsKH5", "name": "Laura", "desc": "Enthusiastic, quirky"},
    "charlie":  {"id": "IKne3meq5aSn9XLyUdCD", "name": "Charlie", "desc": "Deep, confident"},
    "george":   {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "desc": "Warm storyteller"},
    "river":    {"id": "SAz9YHcvj6GT2YYXdXww", "name": "River", "desc": "Relaxed, neutral"},
    "liam":     {"id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam", "desc": "Energetic, social"},
    "alice":    {"id": "Xb7hH8MSUJpSbSDYk0k2", "name": "Alice", "desc": "Clear, engaging"},
    "matilda":  {"id": "XrExE9yKIg1WjnnlVkGX", "name": "Matilda", "desc": "Professional"},
    "jessica":  {"id": "cgSgspJ2msm6clMCkdW9", "name": "Jessica", "desc": "Playful, warm"},
    "eric":     {"id": "cjVigY5qzO86Huf0OWal", "name": "Eric", "desc": "Smooth, trustworthy"},
    "bella":    {"id": "hpp4J3VqNfWAUOO0d1Us", "name": "Bella", "desc": "Professional, bright"},
}

# ── Kokoro (offline fallback) ───────────────────────────────────────────────

MODEL_DIR = str(DATA_DIR / "models")
KOKORO_MODEL = os.path.join(MODEL_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(MODEL_DIR, "voices-v1.0.bin")

KOKORO_VOICE_LIST = {
    "af_heart":  {"name": "Heart", "gender": "F", "desc": "Warm, friendly"},
    "af_alloy":  {"name": "Alloy", "gender": "F", "desc": "Clear, professional"},
    "af_bella":  {"name": "Bella", "gender": "F", "desc": "Confident, bright"},
    "af_nicole": {"name": "Nicole", "gender": "F", "desc": "Smooth, articulate"},
    "af_nova":   {"name": "Nova", "gender": "F", "desc": "Energetic, youthful"},
    "af_sky":    {"name": "Sky", "gender": "F", "desc": "Light, airy"},
    "am_adam":   {"name": "Adam", "gender": "M", "desc": "Deep, authoritative"},
    "am_eric":   {"name": "Eric", "gender": "M", "desc": "Friendly, approachable"},
    "am_michael": {"name": "Michael", "gender": "M", "desc": "Professional, steady"},
    "bf_emma":   {"name": "Emma", "gender": "F", "desc": "Elegant, refined"},
    "bm_daniel": {"name": "Daniel", "gender": "M", "desc": "Classic, distinguished"},
    "bm_george": {"name": "George", "gender": "M", "desc": "Warm, trustworthy"},
}

_kokoro = None
_kokoro_lock = threading.Lock()
_playing = False

PREVIEW_TEXT = "Hi, I'm your voice assistant. How can I help you today?"


def _get_elevenlabs_key():
    """Get ElevenLabs key from env, keychain, or config."""
    key = ELEVENLABS_KEY
    if not key:
        try:
            from config import _keychain_get
            key = _keychain_get("Muse", "elevenlabs_api_key")
        except Exception:
            pass
    return key


def _speak_elevenlabs(text: str, voice: str = "sarah") -> bool:
    """Speak using ElevenLabs API. Returns True on success."""
    key = _get_elevenlabs_key()
    if not key:
        return False
    try:
        import elevenlabs
        elevenlabs.set_api_key(key)
        voice_id = ELEVENLABS_VOICES.get(voice, {}).get("id", voice)
        audio = elevenlabs.generate(text=text, voice=voice_id, model="eleven_turbo_v2_5")
        # Play the audio bytes
        audio_bytes = b"".join(audio) if hasattr(audio, '__iter__') and not isinstance(audio, bytes) else audio
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        # Use afplay for mp3 (simpler than decoding)
        subprocess.run(["afplay", tmp], capture_output=True, timeout=30)
        os.remove(tmp)
        return True
    except Exception as e:
        log.warning("ElevenLabs failed: %s", e)
        return False


def _get_kokoro():
    """Lazy-load Kokoro TTS."""
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    with _kokoro_lock:
        if _kokoro is not None:
            return _kokoro
        if not os.path.exists(KOKORO_MODEL) or not os.path.exists(KOKORO_VOICES):
            return None
        try:
            from kokoro_onnx import Kokoro
            _kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
            log.info("Kokoro TTS loaded")
            return _kokoro
        except Exception as e:
            log.error("Kokoro load failed: %s", e)
            return None


def _speak_kokoro(text: str, voice: str = "af_heart") -> bool:
    """Speak using Kokoro (local). Returns True on success."""
    kokoro = _get_kokoro()
    if not kokoro:
        return False
    try:
        audio, sr = kokoro.create(text, voice=voice, speed=1.0)
        if audio is not None and len(audio) > 0:
            try:
                sd.play(audio, samplerate=sr)
                sd.wait()
            except Exception:
                sd.play(audio, samplerate=sr, device=sd.default.device[1])
                sd.wait()
        return True
    except Exception as e:
        log.warning("Kokoro failed: %s", e)
        return False


def _speak_macos(text: str):
    """Fallback: macOS say command."""
    try:
        subprocess.Popen(["say", "-v", "Samantha", "-r", "190", text],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── Public API ──────────────────────────────────────────────────────────────

def speak(text: str, voice: str = "sarah"):
    """Speak text. ElevenLabs → Kokoro → macOS say. Non-blocking."""
    global _playing
    if _playing:
        stop()

    def run():
        global _playing
        _playing = True
        try:
            # Try ElevenLabs first
            if _speak_elevenlabs(text, voice):
                return
            # Fallback to Kokoro
            kokoro_voice = voice if voice in KOKORO_VOICE_LIST else "af_heart"
            if _speak_kokoro(text, kokoro_voice):
                return
            # Last resort: macOS say
            _speak_macos(text)
        finally:
            _playing = False

    threading.Thread(target=run, daemon=True).start()


def speak_sync(text: str, voice: str = "sarah"):
    """Speak text synchronously (for preview). Blocking."""
    if _speak_elevenlabs(text, voice):
        return
    kokoro_voice = voice if voice in KOKORO_VOICE_LIST else "af_heart"
    if _speak_kokoro(text, kokoro_voice):
        return
    _speak_macos(text)


def stop():
    """Stop playback."""
    global _playing
    try:
        sd.stop()
        subprocess.run(["killall", "afplay"], capture_output=True, timeout=2)
    except Exception:
        pass
    _playing = False


def get_voices() -> dict:
    """Return all available voices grouped by provider."""
    voices = {}
    # ElevenLabs voices (if configured)
    if _get_elevenlabs_key():
        for k, v in ELEVENLABS_VOICES.items():
            voices[k] = {**v, "provider": "elevenlabs"}
    # Kokoro voices (always available offline)
    for k, v in KOKORO_VOICE_LIST.items():
        voices[k] = {**v, "provider": "kokoro"}
    return voices


def is_available() -> bool:
    """Check if any TTS is available."""
    return bool(_get_elevenlabs_key()) or (os.path.exists(KOKORO_MODEL) and os.path.exists(KOKORO_VOICES))
