"""Centralized logging for the voice agent.

Logs to both stderr (when running in terminal) and a rotating file at:
  ~/Library/Logs/VoiceAgent/voice-agent.log
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.expanduser("~/Library/Logs/VoiceAgent")
LOG_FILE = os.path.join(LOG_DIR, "voice-agent.log")
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

_initialized = False


def setup_logging(level: int = logging.INFO):
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler (always)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Stderr handler (only if attached to a terminal)
    if sys.stderr.isatty():
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        root.addHandler(sh)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
