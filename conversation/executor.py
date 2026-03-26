"""Structured tool execution with retries, validation, and fallback chains."""

import asyncio
import random
import subprocess
import time
from dataclasses import dataclass
from logger import get_logger

log = get_logger("executor")


@dataclass
class ToolResult:
    ok: bool
    result: str = ""
    error: str = ""
    attempts: int = 1
    latency_ms: float = 0


class ToolExecutor:
    def __init__(self, desktop, metrics=None):
        self.desktop = desktop
        self.metrics = metrics

    async def execute(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a tool. One attempt, no over-validation."""
        t0 = time.time()
        try:
            raw = await asyncio.to_thread(self._call_sync, tool_name, args)
            ok = not raw.startswith("Error:")
            result = ToolResult(ok=ok, result=raw, latency_ms=(time.time() - t0) * 1000)
            if not ok:
                result.error = raw
            self._record(tool_name, result)
            return result
        except Exception as e:
            result = ToolResult(ok=False, error=str(e), latency_ms=(time.time() - t0) * 1000)
            self._record(tool_name, result)
            return result

    def _call_sync(self, tool: str, args: dict) -> str:
        """Dispatch to desktop method (runs in thread)."""
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
        return f"Unknown tool: {tool}"

    def _record(self, tool: str, result: ToolResult):
        if self.metrics:
            self.metrics.record("tool_execution", result.latency_ms, {
                "tool": tool, "ok": result.ok,
            })
        level = "info" if result.ok else "warning"
        getattr(log, level)("Tool %s: ok=%s, %.0fms", tool, result.ok, result.latency_ms)
