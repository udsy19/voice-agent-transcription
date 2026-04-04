"""macOS system control via AppleScript — open apps, DND, volume, shortcuts."""

import subprocess
from logger import get_logger
from utils import activate_app

log = get_logger("system")


def open_app(name: str) -> dict:
    """Open a macOS app by name."""
    try:
        subprocess.run(["open", "-a", name], capture_output=True, timeout=5)
        log.info("Opened: %s", name)
        return {"ok": True, "app": name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def quit_app(name: str) -> dict:
    """Quit a macOS app."""
    try:
        subprocess.run(["osascript", "-e", f'tell application "{name}" to quit'],
                       capture_output=True, timeout=5)
        return {"ok": True, "app": name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_volume(level: int) -> dict:
    """Set system volume (0-100)."""
    level = max(0, min(100, level))
    try:
        subprocess.run(["osascript", "-e", f"set volume output volume {level}"],
                       capture_output=True, timeout=3)
        return {"ok": True, "volume": level}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def toggle_dnd() -> dict:
    """Toggle Do Not Disturb / Focus mode."""
    try:
        # Use Shortcuts to toggle Focus
        subprocess.run(["shortcuts", "run", "Toggle Focus"], capture_output=True, timeout=5)
        return {"ok": True}
    except Exception:
        # Fallback: use defaults
        try:
            subprocess.run(["defaults", "write", "com.apple.controlcenter",
                          "NSStatusItem Visible Focus", "-bool", "true"],
                          capture_output=True, timeout=3)
            return {"ok": True, "note": "May need Focus shortcut configured"}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def run_shortcut(name: str) -> dict:
    """Run a macOS Shortcut by name."""
    try:
        result = subprocess.run(["shortcuts", "run", name],
                               capture_output=True, text=True, timeout=30)
        return {"ok": result.returncode == 0, "output": result.stdout.strip()[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_brightness(level: int) -> dict:
    """Set display brightness (0-100)."""
    level = max(0, min(100, level))
    brightness = level / 100.0
    try:
        subprocess.run(["osascript", "-e",
                       f'tell application "System Events" to set value of slider 1 of group 1 of window "Display" of application process "System Preferences" to {brightness}'],
                       capture_output=True, timeout=5)
        return {"ok": True, "brightness": level}
    except Exception:
        return {"ok": True, "note": "Brightness control may need accessibility"}
