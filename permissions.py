"""macOS permissions checker — detect, request, and guide users through required permissions.

Each permission has:
- check(): returns "granted" | "denied" | "unknown" | "not_asked"
- instructions: step-by-step guide to enable it
- required: whether the app can't function without it
"""

import os
import sys
import subprocess
from logger import get_logger

log = get_logger("permissions")

# ── Permission definitions ────────────────────────────────────────────────

PERMISSIONS = {
    "microphone": {
        "name": "Microphone",
        "why": "Voice recording for dictation and assistant commands",
        "required": True,
        "instructions": [
            "Open System Settings",
            "Go to Privacy & Security → Microphone",
            "Find Terminal (or Muse) in the list",
            "Toggle it ON",
            "Restart Muse",
        ],
    },
    "input_monitoring": {
        "name": "Input Monitoring",
        "why": "Global hotkeys (⌥L for dictation, ⌥R for assistant)",
        "required": True,
        "instructions": [
            "Open System Settings",
            "Go to Privacy & Security → Input Monitoring",
            "Click the + button",
            "Add Terminal (or Muse.app)",
            "Toggle it ON",
            "Restart Muse",
        ],
    },
    "accessibility": {
        "name": "Accessibility",
        "why": "Pasting transcribed text into apps and system control",
        "required": True,
        "instructions": [
            "Open System Settings",
            "Go to Privacy & Security → Accessibility",
            "Click the + button",
            "Add Terminal (or Muse.app)",
            "Toggle it ON",
            "Restart Muse",
        ],
    },
    "full_disk_access": {
        "name": "Full Disk Access",
        "why": "Reading iMessage history from Messages database",
        "required": False,
        "instructions": [
            "Open System Settings",
            "Go to Privacy & Security → Full Disk Access",
            "Click the + button",
            "Add Terminal (or Muse.app)",
            "Toggle it ON",
            "Restart Muse",
        ],
    },
    "contacts": {
        "name": "Contacts",
        "why": "Looking up contact names for iMessage",
        "required": False,
        "instructions": [
            "Open System Settings",
            "Go to Privacy & Security → Contacts",
            "Find Terminal (or Muse) in the list",
            "Toggle it ON",
        ],
    },
    "location": {
        "name": "Location Services",
        "why": "Location-aware features (weather, nearby search)",
        "required": False,
        "instructions": [
            "Open System Settings",
            "Go to Privacy & Security → Location Services",
            "Make sure Location Services is ON at the top",
            "Scroll down, find Terminal (or Muse)",
            "Toggle it ON",
        ],
    },
    "screen_recording": {
        "name": "Screen Recording",
        "why": "Screenshot analysis ('what's on my screen?')",
        "required": False,
        "instructions": [
            "Open System Settings",
            "Go to Privacy & Security → Screen Recording",
            "Click the + button",
            "Add Terminal (or Muse.app)",
            "Toggle it ON",
            "Restart Muse",
        ],
    },
}


# ── Check functions ───────────────────────────────────────────────────────

def check_microphone() -> str:
    """Check microphone permission. Returns granted/denied/not_asked."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
        if status == 3:
            return "granted"
        elif status == 2:
            return "denied"
        elif status == 0:
            return "not_asked"
        return "denied"  # restricted
    except ImportError:
        # AVFoundation not available — try functional test
        try:
            import sounddevice as sd
            sd.query_devices(kind="input")
            return "unknown"
        except Exception:
            return "denied"


def check_input_monitoring() -> str:
    """Check Input Monitoring / Accessibility trust."""
    try:
        from ApplicationServices import AXIsProcessTrusted
        return "granted" if AXIsProcessTrusted() else "denied"
    except ImportError:
        pass
    try:
        from Quartz import AXIsProcessTrusted
        return "granted" if AXIsProcessTrusted() else "denied"
    except ImportError:
        pass
    # Fallback: try running osascript
    try:
        r = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to return name of first process whose frontmost is true'],
            capture_output=True, text=True, timeout=3)
        return "granted" if r.returncode == 0 else "denied"
    except Exception:
        return "unknown"


def check_accessibility() -> str:
    """Check Accessibility permission (same as input monitoring on macOS)."""
    return check_input_monitoring()


def check_full_disk_access() -> str:
    """Check Full Disk Access by trying to read iMessage db."""
    db = os.path.expanduser("~/Library/Messages/chat.db")
    if not os.path.exists(db):
        return "unknown"  # db doesn't exist (not an Apple device?)
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM message LIMIT 1")
        conn.close()
        return "granted"
    except Exception as e:
        if "unable to open" in str(e).lower() or "authorization denied" in str(e).lower():
            return "denied"
        return "unknown"


def check_contacts() -> str:
    """Check Contacts permission."""
    try:
        import Contacts
        store = Contacts.CNContactStore.alloc().init()
        status = Contacts.CNContactStore.authorizationStatusForEntityType_(0)
        if status == 3:  # authorized
            return "granted"
        elif status == 2:  # denied
            return "denied"
        elif status == 0:  # not determined
            return "not_asked"
        return "denied"
    except ImportError:
        return "unknown"
    except Exception:
        return "unknown"


def check_location() -> str:
    """Check Location Services permission."""
    try:
        import CoreLocation
        status = CoreLocation.CLLocationManager.authorizationStatus()
        if status == 3 or status == 4:  # authorizedAlways or authorizedWhenInUse
            return "granted"
        elif status == 2:
            return "denied"
        elif status == 0:
            return "not_asked"
        return "denied"
    except ImportError:
        return "unknown"
    except Exception:
        return "unknown"


def check_screen_recording() -> str:
    """Check Screen Recording permission by attempting a screenshot."""
    try:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "muse_perm_check.png")
        r = subprocess.run(["screencapture", "-x", "-C", tmp],
                          capture_output=True, timeout=3)
        if os.path.exists(tmp):
            size = os.path.getsize(tmp)
            os.remove(tmp)
            # A denied screenshot produces a tiny file or empty
            return "granted" if size > 500 else "denied"
        return "denied"
    except Exception:
        return "unknown"


# ── Map permission names to check functions ───────────────────────────────

_CHECKERS = {
    "microphone": check_microphone,
    "input_monitoring": check_input_monitoring,
    "accessibility": check_accessibility,
    "full_disk_access": check_full_disk_access,
    "contacts": check_contacts,
    "location": check_location,
    "screen_recording": check_screen_recording,
}


# ── Public API ────────────────────────────────────────────────────────────

def check_all() -> dict:
    """Check all permissions. Returns dict of {name: {status, required, instructions, ...}}."""
    results = {}
    for key, info in PERMISSIONS.items():
        checker = _CHECKERS.get(key)
        status = checker() if checker else "unknown"
        results[key] = {
            "name": info["name"],
            "status": status,
            "required": info["required"],
            "why": info["why"],
            "ok": status in ("granted", "unknown"),
        }
        if status not in ("granted", "unknown"):
            results[key]["instructions"] = info["instructions"]
    return results


def check_required() -> tuple[bool, list[dict]]:
    """Check only required permissions. Returns (all_ok, list_of_issues)."""
    issues = []
    all_perms = check_all()
    for key, info in all_perms.items():
        if info["required"] and not info["ok"]:
            issues.append({
                "permission": info["name"],
                "status": info["status"],
                "why": info["why"],
                "instructions": PERMISSIONS[key]["instructions"],
            })
    return len(issues) == 0, issues


def request_microphone():
    """Request microphone permission (triggers system dialog)."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
        if status == 0:  # not determined
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeAudio, lambda granted: log.info("Mic permission: %s", "granted" if granted else "denied"))
            return True
        return False
    except Exception:
        return False


def print_status():
    """Print permission status to console (for CLI startup)."""
    results = check_all()
    has_issues = False

    for key, info in results.items():
        status = info["status"]
        required = info["required"]
        icon = "✓" if info["ok"] else ("✗" if required else "○")
        label = "REQUIRED" if required and not info["ok"] else ""
        print(f"  {icon} {info['name']}: {status} {label}")
        if not info["ok"] and required:
            has_issues = True

    if has_issues:
        print()
        print("  Some required permissions are missing!")
        print("  Fix them in System Settings > Privacy & Security")
        print()
        for key, info in results.items():
            if not info["ok"] and info["required"]:
                print(f"  === {info['name']} ===")
                for i, step in enumerate(PERMISSIONS[key]["instructions"], 1):
                    print(f"    {i}. {step}")
                print()

    return not has_issues
