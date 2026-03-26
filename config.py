import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix SSL on macOS Python 3.13
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent

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

load_dotenv(DATA_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

# ── API Keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Whisper ──────────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "distil-large-v3")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "auto"

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
SILENCE_THRESHOLD = 0.002
MIN_AUDIO_DURATION = 0.5

# ── Conversation ─────────────────────────────────────────────────────────────
CONVERSATION_MODEL = os.getenv("CONVERSATION_MODEL", "claude-sonnet-4-20250514")
TTS_VOICE = os.getenv("TTS_VOICE", "af_heart")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.3"))
VAD_STOP_SECS = 0.8
SCREENPIPE_URL = "http://localhost:3030"
PORT = int(os.getenv("VOICE_AGENT_PORT", "8528"))
