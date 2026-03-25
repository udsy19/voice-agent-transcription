"""Screen context capture: screenshot + active app + window title.

Captures in parallel with transcription so it adds zero latency.
"""

import base64
import io
import subprocess
from logger import get_logger

log = get_logger("context")

SCREENSHOT_WIDTH = 1280


class ScreenContext:
    def capture(self) -> dict:
        """Capture current screen context."""
        result = {
            "screenshot_b64": "",
            "active_app": "",
            "window_title": "",
            "timestamp": 0,
        }

        # Screenshot
        try:
            result["screenshot_b64"] = self._take_screenshot()
        except Exception as e:
            log.debug("Screenshot failed: %s", e)

        # Active app + window title
        try:
            result["active_app"] = self._get_active_app()
        except Exception:
            pass
        try:
            result["window_title"] = self._get_window_title()
        except Exception:
            pass

        import time
        result["timestamp"] = time.time()
        return result

    def _take_screenshot(self) -> str:
        """Capture screen, resize, return base64 PNG."""
        import mss
        from PIL import Image

        with mss.mss() as sct:
            monitor = sct.monitors[1]  # primary monitor
            img = sct.grab(monitor)

            # Convert to PIL Image and resize
            pil_img = Image.frombytes("RGB", img.size, img.rgb)
            ratio = SCREENSHOT_WIDTH / pil_img.width
            new_size = (SCREENSHOT_WIDTH, int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

            # Encode to base64 PNG
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _get_active_app(self) -> str:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip()

    def _get_window_title(self) -> str:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get title of front window of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip()
