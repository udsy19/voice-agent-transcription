"""Desktop control: AppleScript, keyboard, mouse, Screenpipe queries."""

import subprocess
from logger import get_logger
from injector import inject_text

log = get_logger("desktop")

SCREENPIPE_URL = "http://localhost:3030"


class DesktopController:
    def run_applescript(self, script: str) -> str:
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"Error: {result.stderr.strip()}"
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Error: script timed out"
        except Exception as e:
            return f"Error: {e}"

    def open_app(self, app_name: str) -> str:
        return self.run_applescript(f'tell application "{app_name}" to activate')

    def type_text(self, text: str) -> str:
        success = inject_text(text)
        return "typed successfully" if success else "typing failed"

    def press_shortcut(self, *keys) -> str:
        try:
            import pyautogui
            pyautogui.hotkey(*keys)
            return "shortcut pressed"
        except Exception as e:
            return f"Error: {e}"

    def click(self, x: int, y: int) -> str:
        try:
            import pyautogui
            pyautogui.click(x, y)
            return f"clicked at ({x}, {y})"
        except Exception as e:
            return f"Error: {e}"

    def query_screenpipe(self, query: str, minutes_back: int = 60) -> str:
        """Search recent screen/audio content via Screenpipe API."""
        try:
            import urllib.request
            import json
            url = f"{SCREENPIPE_URL}/search?q={query}&limit=5&content_type=all"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                results = data.get("data", [])
                if not results:
                    return "No relevant screen activity found."
                # Format results
                lines = []
                for r in results[:5]:
                    content = r.get("content", {})
                    text = content.get("text", "")[:200]
                    app = content.get("app_name", "")
                    lines.append(f"[{app}] {text}")
                return "\n".join(lines)
        except Exception:
            return "Screenpipe not available."
