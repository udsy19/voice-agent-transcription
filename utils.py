"""Shared utilities — used across multiple modules."""

import subprocess
import time
import threading
from logger import get_logger

log = get_logger("utils")

# ── Cached active app (osascript is slow ~150-300ms) ────────────────────────
_active_app_cache = {"app": "", "ts": 0.0}
_active_app_lock = threading.Lock()
_ACTIVE_APP_TTL = 2.0  # seconds


def get_active_app() -> str:
    """Get the name of the frontmost application via AppleScript. Cached for 2s."""
    now = time.time()
    with _active_app_lock:
        if now - _active_app_cache["ts"] < _ACTIVE_APP_TTL and _active_app_cache["app"]:
            return _active_app_cache["app"]
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        app = result.stdout.strip()
        with _active_app_lock:
            _active_app_cache["app"] = app
            _active_app_cache["ts"] = now
        return app
    except Exception:
        return _active_app_cache.get("app", "")


def invalidate_active_app_cache():
    """Clear active app cache — call when window focus is known to have changed."""
    with _active_app_lock:
        _active_app_cache["ts"] = 0.0


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


# ── Keychain (cached — `security` subprocess is 500-1000ms each) ────────────

_keychain_cache = {}  # key: (service, account) → value
_keychain_lock = threading.Lock()


def keychain_get(service: str, account: str) -> str:
    """Read from macOS Keychain. Cached in-process for fast repeat access."""
    key = (service, account)
    with _keychain_lock:
        if key in _keychain_cache:
            return _keychain_cache[key]
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        val = r.stdout.strip() if r.returncode == 0 else ""
        with _keychain_lock:
            _keychain_cache[key] = val
        return val
    except Exception:
        return ""


def keychain_set(service: str, account: str, password: str) -> bool:
    """Write to macOS Keychain."""
    try:
        subprocess.run(["security", "delete-generic-password", "-s", service, "-a", account],
                       capture_output=True, timeout=5)
        r = subprocess.run(["security", "add-generic-password", "-s", service, "-a", account, "-w", password],
                           capture_output=True, timeout=5)
        if r.returncode == 0:
            with _keychain_lock:
                _keychain_cache[(service, account)] = password
        return r.returncode == 0
    except Exception:
        return False


def keychain_delete(service: str, account: str) -> bool:
    """Delete from macOS Keychain."""
    with _keychain_lock:
        _keychain_cache.pop((service, account), None)
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


_browser_url_cache = {"url": "", "ts": 0.0}
_browser_url_lock = threading.Lock()
_BROWSER_URL_TTL = 30.0  # seconds — meeting URLs don't change often


def get_browser_url():
    """Get the URL from the frontmost browser tab (Chrome or Safari). Cached 30s."""
    now = time.time()
    with _browser_url_lock:
        if now - _browser_url_cache["ts"] < _BROWSER_URL_TTL:
            return _browser_url_cache["url"]
    # Only check the currently-active browser, not both
    active = get_active_app()
    script = None
    if "Chrome" in active:
        script = 'tell application "Google Chrome" to get URL of active tab of front window'
    elif "Safari" in active:
        script = 'tell application "Safari" to get URL of front document'
    url = ""
    if script:
        try:
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                url = r.stdout.strip()
        except Exception:
            pass
    with _browser_url_lock:
        _browser_url_cache["url"] = url
        _browser_url_cache["ts"] = now
    return url
