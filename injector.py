import subprocess
import time
from logger import get_logger

log = get_logger("injector")

# Typing mode: "paste" (clipboard + Cmd+V) or "type" (character-by-character)
TYPING_MODE = "paste"


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
    """Type text character-by-character using CGEvents.

    Looks like natural typing. Works in apps that don't support paste.
    """
    try:
        from Quartz import (
            CGEventCreateKeyboardEvent, CGEventSetFlags, CGEventPost,
            CGEventKeyboardSetUnicodeString, kCGHIDEventTap,
        )
        import CoreFoundation

        for char in text:
            # Create a key event and set its unicode character
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
        log.error("Character typing failed: %s, falling back to paste", e)
        return False


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


def inject_text(text: str, mode: str | None = None) -> bool:
    """Inject text at cursor. Returns True if successful.

    Args:
        text: Text to inject
        mode: "paste" (clipboard+Cmd+V), "type" (char-by-char), or None (use default)
    """
    if not text:
        return False

    use_mode = mode or TYPING_MODE

    if use_mode == "type":
        # Character-by-character typing
        success = _type_characters(text)
        if success:
            # Also copy to clipboard as backup
            try:
                subprocess.run(["pbcopy"], input=text, text=True, timeout=3, capture_output=True)
            except Exception:
                pass
            return True
        # Fall through to paste mode on failure

    # Paste mode (default)
    try:
        result = subprocess.run(["pbcopy"], input=text, text=True, timeout=3, capture_output=True)
        if result.returncode != 0:
            log.error("pbcopy failed: %s", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        log.error("pbcopy timed out")
        return False
    except FileNotFoundError:
        log.error("pbcopy not found")
        return False
    except Exception as e:
        log.error("pbcopy error: %s", e)
        return False

    time.sleep(0.1)
    success = _press_cmd_v()
    if not success:
        log.warning("Paste failed — text is on clipboard for manual Cmd+V")

    log.info("Injected %d chars via %s", len(text), use_mode)
    return success


def undo_last_paste() -> bool:
    """Send Cmd+Z to undo the last paste."""
    return _press_cmd_z()
