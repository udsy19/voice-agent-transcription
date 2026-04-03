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

def _keychain_get(service: str, account: str) -> str:
    """Read a secret from macOS Keychain. Returns empty string on failure."""
    try:
        import subprocess
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _keychain_set(service: str, account: str, password: str) -> bool:
    """Store a secret in macOS Keychain. Returns True on success."""
    try:
        import subprocess
        # Delete existing entry first (ignore errors)
        subprocess.run(
            ["security", "delete-generic-password", "-s", service, "-a", account],
            capture_output=True, timeout=5,
        )
        result = subprocess.run(
            ["security", "add-generic-password", "-s", service, "-a", account, "-w", password],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# Try Keychain first, then .env fallback
GROQ_API_KEY = _keychain_get("Muse", "groq_api_key") or os.getenv("GROQ_API_KEY", "")
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
