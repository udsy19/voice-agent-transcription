"""Conversation context tracker.

Tracks recent dictations per app to provide multi-turn context to the LLM cleaner.
This helps the cleaner understand pronouns, references, and continuing thoughts.

Example: If you dictated "I'm working on the voice agent project" and then
"it needs better error handling", the cleaner can resolve "it" = "the voice agent project".
"""

import time
from collections import defaultdict
from logger import get_logger

log = get_logger("conversation")

MAX_TURNS = 10          # Keep last N turns per app
MAX_AGE_SEC = 300       # Expire turns older than 5 minutes
MAX_CONTEXT_CHARS = 600 # Cap context string length sent to LLM


class ConversationTracker:
    """Tracks recent dictation turns per application for context continuity."""

    def __init__(self):
        # {app_name: [{"text": str, "ts": float}, ...]}
        self._turns: dict[str, list[dict]] = defaultdict(list)

    def add_turn(self, app_name: str, text: str):
        """Record a completed dictation turn."""
        if not text or not text.strip():
            return
        app = app_name or "_global"
        self._turns[app].append({
            "text": text.strip(),
            "ts": time.time(),
        })
        # Cap turns per app
        if len(self._turns[app]) > MAX_TURNS:
            self._turns[app] = self._turns[app][-MAX_TURNS:]

    def get_context(self, app_name: str) -> str | None:
        """Get recent conversation context for an app.

        Returns a formatted string of recent turns, or None if no context.
        Only includes turns from the last 5 minutes.
        """
        app = app_name or "_global"
        turns = self._turns.get(app, [])
        if not turns:
            return None

        now = time.time()
        recent = [t for t in turns if (now - t["ts"]) < MAX_AGE_SEC]
        if not recent:
            return None

        # Build context string, respecting character limit
        lines = []
        total = 0
        for turn in reversed(recent):  # most recent first for truncation
            line = turn["text"]
            if total + len(line) > MAX_CONTEXT_CHARS:
                break
            lines.append(line)
            total += len(line)

        if not lines:
            return None

        lines.reverse()  # back to chronological order
        context = "Recent dictations in this app:\n" + "\n".join(f"- {l}" for l in lines)
        log.debug("Context for %s: %d turns, %d chars", app, len(lines), len(context))
        return context

    def clear(self, app_name: str | None = None):
        """Clear conversation history for an app (or all apps)."""
        if app_name:
            self._turns.pop(app_name, None)
            self._turns.pop("_global", None)
        else:
            self._turns.clear()

    def prune(self):
        """Remove expired turns across all apps."""
        now = time.time()
        for app in list(self._turns.keys()):
            self._turns[app] = [t for t in self._turns[app] if (now - t["ts"]) < MAX_AGE_SEC]
            if not self._turns[app]:
                del self._turns[app]
