"""Shared utilities — used across multiple modules."""

import subprocess
from logger import get_logger

log = get_logger("utils")


def get_active_app() -> str:
    """Get the name of the frontmost application via AppleScript."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _escape_applescript(s: str) -> str:
    """Escape a string for safe embedding in AppleScript double-quoted strings."""
    return s.replace('\\', '\\\\').replace('"', '\\"')


def activate_app(app_name: str) -> bool:
    """Bring an app to front using AppleScript."""
    if not app_name:
        return False
    try:
        safe_name = _escape_applescript(app_name)
        script = f'tell application "System Events" to set frontmost of process "{safe_name}" to true'
        result = subprocess.run(["osascript", "-e", script],
                                capture_output=True, text=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


# ── Keychain ────────────────────────────────────────────────────────────────

def keychain_get(service: str, account: str) -> str:
    """Read from macOS Keychain."""
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def keychain_set(service: str, account: str, password: str) -> bool:
    """Write to macOS Keychain."""
    try:
        subprocess.run(["security", "delete-generic-password", "-s", service, "-a", account],
                       capture_output=True, timeout=5)
        r = subprocess.run(["security", "add-generic-password", "-s", service, "-a", account, "-w", password],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def keychain_delete(service: str, account: str) -> bool:
    """Delete from macOS Keychain."""
    try:
        r = subprocess.run(["security", "delete-generic-password", "-s", service, "-a", account],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# ── Audio Devices ──────────────────────────────────────────────────────────

def detect_blackhole():
    """Return BlackHole device index and name if installed, else (None, None)."""
    try:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if "blackhole" in d["name"].lower() and d["max_input_channels"] > 0:
                return i, d["name"]
    except Exception:
        pass
    return None, None


def get_browser_url():
    """Get the URL from the frontmost browser tab (Chrome or Safari)."""
    for app, script in [
        ("Google Chrome", 'tell application "Google Chrome" to get URL of active tab of front window'),
        ("Safari", 'tell application "Safari" to get URL of front document'),
    ]:
        try:
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    return ""
