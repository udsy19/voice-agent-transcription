"""Safe JSON load/save with backup and recovery."""

import json
import os
import shutil
from logger import get_logger

log = get_logger("safe_json")


def load(path: str, default=None):
    """Load JSON with automatic backup of corrupt files.

    If the file is corrupt, backs it up as .corrupt and returns default.
    """
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, ValueError) as e:
        # Back up corrupt file
        backup = path + ".corrupt"
        try:
            shutil.copy2(path, backup)
            log.error("Corrupt JSON at %s — backed up to %s: %s", path, backup, e)
        except Exception:
            log.error("Corrupt JSON at %s: %s", path, e)
        return default
    except IOError as e:
        log.error("Cannot read %s: %s", path, e)
        return default


def save(path: str, data, indent: int = 2) -> bool:
    """Save JSON atomically — writes to temp file first, then renames.

    This prevents corruption if the process crashes mid-write.
    """
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)  # atomic on POSIX
        return True
    except Exception as e:
        log.error("Failed to save %s: %s", path, e)
        try:
            os.remove(tmp)
        except Exception:
            pass
        return False
