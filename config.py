import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project directory (not cwd, which may be /tmp)
_project_dir = Path(__file__).parent
load_dotenv(_project_dir / ".env")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

# Whisper
WHISPER_MODEL = "distil-large-v3"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "auto"

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# Silence detection
SILENCE_THRESHOLD = 0.005  # RMS below this = silence
MIN_AUDIO_DURATION = 0.5  # ignore recordings shorter than this (seconds)
