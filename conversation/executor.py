"""Structured tool execution with retries, validation, and fallback chains.

Every tool call goes through ToolExecutor which:
1. Classifies errors (transient vs permanent)
2. Retries transient errors with exponential backoff
3. Validates results (did it actually work?)
4. Falls back to alternative methods
5. Records metrics
"""

import asyncio
import random
import subprocess
import time
from enum import Enum
from dataclasses import dataclass, field
from logger import get_logger

log = get_logger("executor")


class ErrorType(Enum):
    TRANSIENT = "transient"  # timeout, busy, network — retry
    PERMANENT = "permanent"  # not found, permission denied — don't retry
    VALIDATION = "validation"  # action ran but result wrong — try fallback


@dataclass
class ToolResult:
    ok: bool
    result: str = ""
    error: str = ""
    error_type: ErrorType | None = None
    attempts: int = 1
    latency_ms: float = 0


def _classify_error(e: Exception) -> ErrorType:
    msg = str(e).lower()
    if any(k in msg for k in ["timeout", "timed out", "busy", "connection"]):
        return ErrorType.TRANSIENT
    if any(k in msg for k in ["not found", "permission", "denied", "not authorized"]):
        return ErrorType.PERMANENT
    return ErrorType.TRANSIENT  # default: retry


def _is_app_running(app_name: str) -> bool:
    """Check if an app is actually running."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             f'tell application "System Events" to (name of processes) contains "{app_name}"'],
            capture_output=True, text=True, timeout=3,
        )
        return "true" in result.stdout.lower()
    except Exception:
        return False


class ToolExecutor:
    """Execute tools with retry, validation, and fallback."""

    def __init__(self, desktop, metrics=None):
        self.desktop = desktop
        self.metrics = metrics

    async def execute(self, tool_name: str, args: dict, max_retries: int = 2) -> ToolResult:
        """Execute a tool with structured error handling."""
        t0 = time.time()

        for attempt in range(max_retries + 1):
            try:
                raw = await self._call(tool_name, args)
                result = ToolResult(ok=True, result=raw, attempts=attempt + 1)

                # Validate
                if not self._validate(tool_name, args, raw):
                    if attempt < max_retries:
                        log.warning("Tool %s validation failed, retrying", tool_name)
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    result = ToolResult(
                        ok=False, result=raw, error="Validation failed",
                        error_type=ErrorType.VALIDATION, attempts=attempt + 1,
                    )

                result.latency_ms = (time.time() - t0) * 1000
                self._record(tool_name, result)
                return result

            except Exception as e:
                err_type = _classify_error(e)
                log.warning("Tool %s attempt %d failed (%s): %s", tool_name, attempt + 1, err_type.value, e)

                if err_type == ErrorType.PERMANENT or attempt >= max_retries:
                    # Try fallback
                    fallback = await self._try_fallback(tool_name, args)
                    if fallback:
                        fallback.latency_ms = (time.time() - t0) * 1000
                        self._record(tool_name, fallback)
                        return fallback

                    result = ToolResult(
                        ok=False, error=str(e), error_type=err_type, attempts=attempt + 1,
                        latency_ms=(time.time() - t0) * 1000,
                    )
                    self._record(tool_name, result)
                    return result

                # Exponential backoff with jitter
                wait = (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(wait)

        result = ToolResult(ok=False, error="Max retries exceeded", attempts=max_retries + 1)
        result.latency_ms = (time.time() - t0) * 1000
        self._record(tool_name, result)
        return result

    async def _call(self, tool: str, args: dict) -> str:
        """Dispatch to the appropriate desktop method."""
        if tool == "run_applescript":
            return self.desktop.run_applescript(args["script"])
        elif tool == "read_app_ui":
            return self.desktop.read_ui_tree(args["app_name"])
        elif tool == "click_element":
            return self.desktop.click_element(args["app_name"], args["element_name"])
        elif tool == "type_in_element":
            return self.desktop.type_in_element(args["app_name"], args["element_name"], args["text"])
        elif tool == "type_text":
            return self.desktop.type_text(args["text"])
        elif tool == "open_app":
            return self.desktop.open_app(args["app_name"])
        elif tool == "press_shortcut":
            return self.desktop.press_shortcut(*args["keys"])
        elif tool == "query_screen_memory":
            return self.desktop.query_screenpipe(args["query"], args.get("minutes_back", 60))
        elif tool == "click_at_coordinates":
            return self.desktop.click(args["x"], args["y"])
        else:
            return f"Unknown tool: {tool}"

    def _validate(self, tool: str, args: dict, result: str) -> bool:
        """Check if the tool actually did what we asked."""
        # AppleScript errors always start with specific patterns
        if result.startswith("Error:") or "execution error" in result.lower():
            return False
        if tool == "open_app":
            return _is_app_running(args.get("app_name", ""))
        if tool == "read_app_ui":
            return len(result) > 5
        # For run_applescript: empty result is OK (means script ran without output)
        if tool == "run_applescript":
            return "error:" not in result.lower()[:20]
        return True

    async def _try_fallback(self, tool: str, args: dict) -> ToolResult | None:
        """Try alternative method if primary fails."""
        log.info("Trying fallback for %s", tool)

        if tool == "open_app":
            # Fallback: use `open -a` shell command
            app = args.get("app_name", "")
            try:
                subprocess.run(["open", "-a", app], capture_output=True, timeout=5)
                if _is_app_running(app):
                    return ToolResult(ok=True, result=f"Opened {app} via shell fallback")
            except Exception:
                pass

        elif tool == "click_element":
            # Fallback: try AppleScript click
            app = args.get("app_name", "")
            elem = args.get("element_name", "")
            result = self.desktop.click_element(app, elem)
            if "Error" not in result:
                return ToolResult(ok=True, result=result)

        elif tool == "run_applescript":
            # No good fallback for arbitrary scripts
            pass

        return None

    def _record(self, tool: str, result: ToolResult):
        """Record metrics."""
        if self.metrics:
            self.metrics.record("tool_execution", result.latency_ms, {
                "tool": tool, "ok": result.ok, "attempts": result.attempts,
            })
        log.info("Tool %s: ok=%s, %dms, %d attempts", tool, result.ok, result.latency_ms, result.attempts)
