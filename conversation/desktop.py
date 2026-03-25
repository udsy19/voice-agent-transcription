"""Desktop control — 4-layer execution stack.

Layer 1: AppleScript (native, fast, structured)
Layer 2: Accessibility API via osascript (reads UI tree)
Layer 3: Terminator / desktop-use (80ms UI element interaction)
Layer 4: pyautogui click (last resort)

Also handles instant actions (media, volume, lock, screenshot).
"""

import subprocess
import os
from logger import get_logger
from injector import inject_text, undo_last_paste

log = get_logger("desktop")


class DesktopController:
    """Multi-layer desktop automation."""

    def __init__(self):
        self._terminator = None
        self._init_terminator()

    def _init_terminator(self):
        """Try to load desktop-use client."""
        try:
            from desktop_use import DesktopUseClient
            self._terminator = DesktopUseClient()
            log.info("Terminator (desktop-use) available")
        except Exception as e:
            log.info("Terminator not available: %s — using AppleScript/accessibility", e)

    # ── Instant actions (no LLM needed) ──────────────────────────────────────

    async def execute_instant(self, action: dict) -> str:
        """Execute a tier-1 instant action. Returns result string."""
        act = action.get("action", "")

        if act == "open_app":
            app = action.get("app", "")
            return self.open_app(app)

        elif act == "media":
            ma = action.get("action", "playpause")
            script = {
                "playpause": 'tell application "Music" to playpause',
                "next": 'tell application "Music" to next track',
                "previous": 'tell application "Music" to previous track',
            }.get(ma, 'tell application "Music" to playpause')
            return self.run_applescript(script)

        elif act == "volume":
            d = action.get("direction", "up")
            if d == "mute":
                return self.run_applescript("set volume with output muted")
            elif d == "up":
                return self.run_applescript("set volume output volume ((output volume of (get volume settings)) + 10)")
            else:
                return self.run_applescript("set volume output volume ((output volume of (get volume settings)) - 10)")

        elif act == "screenshot":
            self.run_applescript('do shell script "screencapture -ic"')
            return "Screenshot taken and copied to clipboard"

        elif act == "type":
            text = action.get("text", "")
            inject_text(text)
            return f"Typed: {text[:50]}"

        elif act == "undo":
            undo_last_paste()
            return "Undone"

        elif act == "lock":
            self.run_applescript('tell application "System Events" to keystroke "q" using {command down, control down}')
            return "Screen locked"

        return f"Unknown instant action: {act}"

    # ── Layer 1: AppleScript ─────────────────────────────────────────────────

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
        # Try exact name first, then fuzzy
        result = self.run_applescript(f'tell application "{app_name}" to activate')
        if "Error" not in result:
            return f"Opened {app_name}"
        # Try with .app suffix
        result = self.run_applescript(f'do shell script "open -a \\"{app_name}\\""')
        return f"Opened {app_name}" if "Error" not in result else result

    def type_text(self, text: str) -> str:
        success = inject_text(text)
        return "Typed successfully" if success else "Typing failed — text is on clipboard"

    def press_shortcut(self, *keys) -> str:
        try:
            import pyautogui
            pyautogui.hotkey(*keys)
            return "Shortcut pressed"
        except Exception as e:
            return f"Error: {e}"

    def click(self, x: int, y: int) -> str:
        try:
            import pyautogui
            pyautogui.click(x, y)
            return f"Clicked at ({x}, {y})"
        except Exception as e:
            return f"Error: {e}"

    # ── Layer 2: Accessibility API ───────────────────────────────────────────

    def read_ui_tree(self, app_name: str) -> str:
        """Read the full UI tree of an app via accessibility API."""
        script = f'''
        tell application "System Events"
            tell process "{app_name}"
                set output to ""
                repeat with w in windows
                    set output to output & "WINDOW: " & name of w & return
                    try
                        repeat with e in entire contents of w
                            try
                                set r to role of e
                                set v to value of e
                                set d to description of e
                                set n to name of e
                                if n is not missing value and n is not "" then
                                    set output to output & r & " [" & n & "]"
                                    if v is not missing value and v is not "" then
                                        set output to output & ": " & v
                                    end if
                                    set output to output & return
                                end if
                            end try
                        end repeat
                    end try
                end repeat
                return output
            end tell
        end tell
        '''
        result = self.run_applescript(script)
        # Truncate if too long
        if len(result) > 4000:
            result = result[:4000] + "\n... (truncated)"
        return result

    def click_element(self, app_name: str, element_name: str) -> str:
        """Click a named UI element via accessibility."""
        script = f'''
        tell application "System Events"
            tell process "{app_name}"
                try
                    click button "{element_name}" of window 1
                    return "clicked button"
                on error
                    try
                        click menu item "{element_name}" of menu bar 1
                        return "clicked menu item"
                    on error
                        return "element not found: {element_name}"
                    end try
                end try
            end tell
        end tell
        '''
        return self.run_applescript(script)

    def type_in_element(self, app_name: str, element_name: str, text: str) -> str:
        """Type text into a named UI element."""
        script = f'''
        tell application "System Events"
            tell process "{app_name}"
                try
                    set value of text field "{element_name}" of window 1 to "{text}"
                    return "typed"
                on error
                    try
                        set focused of text field 1 of window 1 to true
                        keystroke "{text}"
                        return "typed via keystroke"
                    on error errMsg
                        return "error: " & errMsg
                    end try
                end try
            end tell
        end tell
        '''
        return self.run_applescript(script)

    # ── Layer 3: Terminator ──────────────────────────────────────────────────

    async def terminator_read_ui(self, app_name: str) -> str:
        """Read UI tree via Terminator (faster than AppleScript)."""
        if not self._terminator:
            return self.read_ui_tree(app_name)  # fallback
        try:
            locator = self._terminator.locator(f"window:{app_name}")
            elements = await locator.get_all_elements()
            return str(elements)[:4000]
        except Exception as e:
            log.debug("Terminator read failed: %s, falling back to AppleScript", e)
            return self.read_ui_tree(app_name)

    async def terminator_click(self, app_name: str, element_name: str) -> str:
        if not self._terminator:
            return self.click_element(app_name, element_name)
        try:
            locator = self._terminator.locator(f"window:{app_name}").locator(f"name:{element_name}")
            await locator.click()
            return f"Clicked {element_name}"
        except Exception as e:
            log.debug("Terminator click failed: %s, falling back", e)
            return self.click_element(app_name, element_name)

    async def terminator_type(self, app_name: str, element_name: str, text: str) -> str:
        if not self._terminator:
            return self.type_in_element(app_name, element_name, text)
        try:
            locator = self._terminator.locator(f"window:{app_name}").locator(f"name:{element_name}")
            await locator.type_text(text)
            return f"Typed in {element_name}"
        except Exception as e:
            log.debug("Terminator type failed: %s, falling back", e)
            return self.type_in_element(app_name, element_name, text)

    # ── Screenpipe ───────────────────────────────────────────────────────────

    def query_screenpipe(self, query: str, minutes_back: int = 60) -> str:
        try:
            import urllib.request
            import json
            url = f"http://localhost:3030/search?q={query}&limit=5&content_type=all"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
                results = data.get("data", [])
                if not results:
                    return "No relevant screen activity found."
                lines = []
                for r in results[:5]:
                    content = r.get("content", {})
                    text = content.get("text", "")[:200]
                    app = content.get("app_name", "")
                    lines.append(f"[{app}] {text}")
                return "\n".join(lines)
        except Exception:
            return "Screenpipe not available."
