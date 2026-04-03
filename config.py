import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────

# Project source directory (where the .py files live)
PROJECT_DIR = Path(__file__).parent

# Persistent data directory (survives restarts, app updates)
# macOS: ~/Library/Application Support/VoiceAgent/
# Linux: ~/.local/share/VoiceAgent/
# Windows: %APPDATA%/VoiceAgent/
# Electron passes this, or we detect the standard location
_env_data_dir = os.environ.get("VOICE_AGENT_DATA_DIR")
if _env_data_dir:
    DATA_DIR = Path(_env_data_dir)
elif sys.platform == "darwin":
    DATA_DIR = Path.home() / "Library" / "Application Support" / "VoiceAgent"
elif sys.platform == "win32":
    DATA_DIR = Path(os.environ.get("APPDATA", "~")) / "VoiceAgent"
else:
    DATA_DIR = Path.home() / ".local" / "share" / "VoiceAgent"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env from project dir OR data dir
load_dotenv(DATA_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

# ── Groq ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Whisper ──────────────────────────────────────────────────────────────────
WHISPER_MODEL = "distil-large-v3"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "auto"

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# ── Silence detection ────────────────────────────────────────────────────────
SILENCE_THRESHOLD = 0.002  # very sensitive — captures faint audio
MIN_AUDIO_DURATION = 0.5
