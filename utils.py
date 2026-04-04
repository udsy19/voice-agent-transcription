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


def activate_app(app_name: str) -> bool:
    """Bring an app to front using AppleScript."""
    if not app_name:
        return False
    try:
        script = f'tell application "System Events" to set frontmost of process "{app_name}" to true'
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
