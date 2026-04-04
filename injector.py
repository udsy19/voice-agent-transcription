import subprocess
import time
from logger import get_logger

log = get_logger("injector")

TYPING_MODE = "paste"


def _set_clipboard(text: str) -> bool:
    """Write text to clipboard via pbcopy."""
    try:
        result = subprocess.run(
            ["pbcopy"], input=text, text=True, timeout=3, capture_output=True
        )
        return result.returncode == 0
    except Exception as e:
        log.error("pbcopy failed: %s", e)
        return False


# ── App focus ───────────────────────────────────────────────────────────────

from utils import get_active_app as _get_frontmost_app, activate_app as _activate_app


def _wait_for_app_focus(app_name: str, timeout: float = 1.0) -> bool:
    """Wait until the given app is actually frontmost. Returns True if focused."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = _get_frontmost_app()
        if current == app_name or app_name.lower() in current.lower():
            return True
        time.sleep(0.05)
    log.warning("Timed out waiting for '%s' to become frontmost", app_name)
    return False


# ── Quartz key simulation ──────────────────────────────────────────────────

def _press_cmd_v():
    """Simulate Cmd+V using Quartz CGEvents."""
    try:
        from Quartz import (
            CGEventCreateKeyboardEvent, CGEventSetFlags, CGEventPost,
            kCGHIDEventTap, kCGEventFlagMaskCommand,
        )
        V_KEY = 9
        time.sleep(0.05)

        event_down = CGEventCreateKeyboardEvent(None, V_KEY, True)
        if event_down is None:
            log.error("Failed to create key-down event")
            return False
        CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event_down)
        time.sleep(0.02)

        event_up = CGEventCreateKeyboardEvent(None, V_KEY, False)
        if event_up is None:
            return False
        CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event_up)
        return True
    except Exception as e:
        log.error("CGEvent failed: %s", e)
        return False


def _press_cmd_z():
    """Simulate Cmd+Z (undo)."""
    try:
        from Quartz import (
            CGEventCreateKeyboardEvent, CGEventSetFlags, CGEventPost,
            kCGHIDEventTap, kCGEventFlagMaskCommand,
        )
        Z_KEY = 6
        event_down = CGEventCreateKeyboardEvent(None, Z_KEY, True)
        if event_down is None:
            return False
        CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event_down)
        time.sleep(0.02)
        event_up = CGEventCreateKeyboardEvent(None, Z_KEY, False)
        if event_up is None:
            return False
        CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event_up)
        return True
    except Exception as e:
        log.error("Undo failed: %s", e)
        return False


def _type_characters(text: str, delay: float = 0.008):
    """Type text character-by-character using CGEvents."""
    try:
        from Quartz import (
            CGEventCreateKeyboardEvent, CGEventPost,
            CGEventKeyboardSetUnicodeString, kCGHIDEventTap,
        )

        for char in text:
            event_down = CGEventCreateKeyboardEvent(None, 0, True)
            if event_down is None:
                continue
            CGEventKeyboardSetUnicodeString(event_down, len(char), char)
            CGEventPost(kCGHIDEventTap, event_down)

            event_up = CGEventCreateKeyboardEvent(None, 0, False)
            if event_up is None:
                continue
            CGEventKeyboardSetUnicodeString(event_up, len(char), char)
            CGEventPost(kCGHIDEventTap, event_up)

            time.sleep(delay)

        log.info("Typed %d chars character-by-character", len(text))
        return True
    except Exception as e:
        log.error("Character typing failed: %s", e)
        return False


# ── Public API ──────────────────────────────────────────────────────────────

def get_selected_text() -> str:
    """Get currently selected text by simulating Cmd+C and reading clipboard."""
    try:
        old = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        old = ""

    try:
        from Quartz import (CGEventCreateKeyboardEvent, CGEventSetFlags,
                            CGEventPost, kCGHIDEventTap, kCGEventFlagMaskCommand)
        C_KEY = 8
        for pressed in (True, False):
            ev = CGEventCreateKeyboardEvent(None, C_KEY, pressed)
            if ev is None:
                return ""
            CGEventSetFlags(ev, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, ev)
    except Exception as e:
        log.error("Cmd+C failed: %s", e)
        return ""

    time.sleep(0.15)

    try:
        selected = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        selected = ""

    if old and old != selected:
        subprocess.run(["pbcopy"], input=old, text=True, timeout=2)

    return selected if selected != old else ""


def inject_text(text: str, target_app: str = "", mode: str | None = None) -> bool:
    """Inject text at cursor in the target app. Returns True if successful.

    Args:
        text: Text to inject
        target_app: App name to focus before pasting (if empty, pastes into current app)
        mode: "paste" (clipboard+Cmd+V), "type" (char-by-char), or None (use default)
    """
    if not text:
        return False

    # Step 1: If target app specified, refocus it first
    if target_app:
        current = _get_frontmost_app()
        if current != target_app and target_app.lower() not in current.lower():
            log.info("Refocusing '%s' (currently: '%s')", target_app, current)
            _activate_app(target_app)
            _wait_for_app_focus(target_app, timeout=3.0)
            time.sleep(0.15)

    use_mode = mode or TYPING_MODE

    if use_mode == "type":
        success = _type_characters(text)
        if success:
            _set_clipboard(text)  # backup on clipboard
            return True
        # Fall through to paste mode on failure

    # Step 2: Paste mode
    if not _set_clipboard(text):
        log.error("Failed to set clipboard")
        return False

    time.sleep(0.1)
    success = _press_cmd_v()
    if not success:
        log.warning("Paste failed — text is on clipboard for manual Cmd+V")

    log.info("Injected %d chars via %s into '%s'", len(text), use_mode, target_app or "current app")
    return success


def undo_last_paste() -> bool:
    """Send Cmd+Z to undo the last paste."""
    return _press_cmd_z()
