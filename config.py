import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────

# Project source directory (where the .py files live)
PROJECT_DIR = Path(__file__).parent

# Persistent data directory (survives restarts, app updates)
# macOS: ~/Library/Application Support/Muse/
# Linux: ~/.local/share/Muse/
# Windows: %APPDATA%/Muse/
# Electron passes this, or we detect the standard location
_env_data_dir = os.environ.get("VOICE_AGENT_DATA_DIR")
if _env_data_dir:
    DATA_DIR = Path(_env_data_dir)
elif sys.platform == "darwin":
    DATA_DIR = Path.home() / "Library" / "Application Support" / "Muse"
elif sys.platform == "win32":
    DATA_DIR = Path(os.environ.get("APPDATA", "~")) / "Muse"
else:
    DATA_DIR = Path.home() / ".local" / "share" / "Muse"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env from project dir OR data dir
load_dotenv(DATA_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

# ── Groq ─────────────────────────────────────────────────────────────────────

from utils import keychain_get as _keychain_get, keychain_set as _keychain_set


# Try Keychain first, then .env fallback
GROQ_API_KEY = _keychain_get("Muse", "groq_api_key") or os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = _keychain_get("Muse", "anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
# ELEVENLABS_API_KEY removed — using Kokoro only
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# LLM provider: "hybrid" (local + cloud), "local" (offline only), "groq" (cloud only)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hybrid")

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
MIN_AUDIO_DURATION = 1.0  # reject < 1s to avoid Whisper hallucinations
