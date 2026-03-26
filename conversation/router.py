"""3-tier intent router with semantic embeddings.

Tier 1: INSTANT — regex match, direct execution (<10ms)
Tier 2: FAST    — Groq, no tools, simple conversation (~400ms)
Tier 3: THINK   — Claude + tools + optional vision (~800ms)

Routing: regex → embedding similarity → keyword fallback
"""

import os
import re
import numpy as np
from logger import get_logger

log = get_logger("router")

# ── Tier 1: Instant patterns ────────────────────────────────────────────────

INSTANT_PATTERNS = [
    (r"\b(?:open|launch|start|switch to)\s+(\w[\w\s]*?)(?:\s+app)?$",
     lambda m: ("open_app", {"app": m.group(1).strip()})),
    (r"\b(?:play|pause|resume)\s*(?:music|song|track)?$",
     lambda _: ("media", {"action": "playpause"})),
    (r"\b(?:next|skip)\s*(?:song|track)?$",
     lambda _: ("media", {"action": "next"})),
    (r"\b(?:previous)\s*(?:song|track)?$",
     lambda _: ("media", {"action": "previous"})),
    (r"\bvolume\s*(up|down)$",
     lambda m: ("volume", {"direction": m.group(1)})),
    (r"\b(?:mute|unmute)$",
     lambda _: ("volume", {"direction": "mute"})),
    (r"\btake\s*(?:a\s+)?screenshot$",
     lambda _: ("screenshot", {})),
    (r"\btype\s+(.+)$",
     lambda m: ("type", {"text": m.group(1)})),
    (r"\bundo\s*(?:that)?$",
     lambda _: ("undo", {})),
    (r"\block\s*(?:the\s+)?(?:screen|computer)$",
     lambda _: ("lock", {})),
]

# ── Semantic intent examples (for embedding-based routing) ───────────────────

INTENT_EXAMPLES = {
    "tool_needed": [
        "check my email", "read my messages", "what emails do I have",
        "schedule a meeting", "what's on my calendar", "send a message",
        "reply to that email", "forward this to", "compose an email",
        "find the file", "search for", "look up", "read the document",
        "text someone", "message them", "call", "contact",
        "remind me", "set a reminder", "create a task",
        "open messages and tell me", "check slack", "read notifications",
    ],
    "vision_needed": [
        "what is this on screen", "what am I looking at",
        "describe what you see", "read this for me",
        "click on that button", "fill out this form",
        "what does this say", "rewrite this", "edit this",
        "look at this", "what's showing", "this page",
    ],
    "simple_chat": [
        "hello", "how are you", "thanks", "goodbye",
        "what time is it", "tell me a joke", "what's the weather",
        "define this word", "translate this", "calculate",
        "who is", "what is", "explain", "how does",
    ],
}

# ── App Registry ─────────────────────────────────────────────────────────────

APP_CATEGORIES = {
    "email": ["Spark", "Mail", "Outlook", "Airmail"],
    "calendar": ["Fantastical", "Calendar", "BusyCal"],
    "browser": ["Arc", "Chrome", "Safari", "Firefox", "Brave"],
    "code": ["Cursor", "Visual Studio Code", "Xcode", "PyCharm", "IntelliJ IDEA"],
    "messages": ["Slack", "WhatsApp", "Telegram", "Messages", "Discord", "Signal"],
    "music": ["Spotify", "Music", "Apple Music"],
    "notes": ["Notion", "Obsidian", "Notes", "Bear"],
    "terminal": ["Terminal", "iTerm", "Warp", "cmux", "Alacritty", "kitty"],
    "files": ["Finder"],
    "design": ["Figma", "Sketch"],
    "video": ["Zoom", "Google Meet", "FaceTime"],
}

APP_MECHANISMS = {
    "Mail": ["applescript"], "Calendar": ["applescript"],
    "Contacts": ["applescript"], "Finder": ["applescript"],
    "Safari": ["applescript", "accessibility"], "Messages": ["applescript"],
    "Notes": ["applescript"], "Reminders": ["applescript"],
    "Music": ["applescript"], "Terminal": ["applescript"],
    "Spark": ["accessibility"], "Slack": ["accessibility", "applescript"],
    "Chrome": ["applescript", "accessibility"], "Arc": ["accessibility"],
    "Notion": ["accessibility"], "Linear": ["accessibility"],
    "VS Code": ["accessibility"], "Cursor": ["accessibility"],
}


class SemanticRouter:
    """Routes transcripts using regex → embeddings → keyword fallback."""

    def __init__(self):
        self._encoder = None
        self._intent_embeddings: dict[str, np.ndarray] = {}
        self._loaded = False

    def _load_encoder(self):
        """Skip heavy embedding model — keyword routing is fast and good enough."""
        self._loaded = True

    def route(self, transcript: str) -> tuple[str, dict | None]:
        """Route transcript to appropriate tier."""
        text = transcript.strip()
        lower = text.lower()

        # Tier 1: instant regex
        for pattern, handler in INSTANT_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                action_type, args = handler(match)
                log.info("Tier 1 (instant): %s → %s", lower[:40], action_type)
                return "instant", {"action": action_type, **args}

        # Tier 2/3: semantic routing
        self._load_encoder()
        if self._encoder and self._intent_embeddings:
            intent, confidence = self._semantic_route(text)
            if confidence > 0.55:
                if intent == "tool_needed":
                    log.info("Tier 3 (claude, semantic %.0f%%): %s", confidence * 100, lower[:40])
                    return "claude", None
                elif intent == "vision_needed":
                    log.info("Tier 3 (claude+vision, semantic %.0f%%): %s", confidence * 100, lower[:40])
                    return "claude", None
                elif intent == "simple_chat":
                    log.info("Tier 2 (groq, semantic %.0f%%): %s", confidence * 100, lower[:40])
                    return "groq", None

        # Keyword fallback
        tool_kw = {"email", "mail", "inbox", "calendar", "schedule", "meeting",
                   "message", "send", "reply", "check", "search", "find", "read",
                   "remind", "task", "slack", "text", "contact", "compose"}
        if any(kw in lower for kw in tool_kw):
            log.info("Tier 3 (claude, keyword): %s", lower[:40])
            return "claude", None

        vision_kw = {"this", "that", "here", "screen", "what's on", "look at",
                     "see", "showing", "button", "click", "fill", "form", "rewrite", "edit this"}
        if any(kw in lower for kw in vision_kw):
            log.info("Tier 3 (claude+vision, keyword): %s", lower[:40])
            return "claude", None

        # Default: short = groq, long = claude
        if len(text.split()) <= 8:
            log.info("Tier 2 (groq, short): %s", lower[:40])
            return "groq", None
        log.info("Tier 3 (claude): %s", lower[:40])
        return "claude", None

    def _semantic_route(self, text: str) -> tuple[str, float]:
        """Route via cosine similarity to intent examples."""
        query = self._encoder.encode(text).reshape(1, -1)
        best_intent = "simple_chat"
        best_score = 0.0

        for intent, embeddings in self._intent_embeddings.items():
            scores = np.dot(query, embeddings.T)[0]
            max_score = float(np.max(scores))
            if max_score > best_score:
                best_score = max_score
                best_intent = intent

        return best_intent, best_score


# Global router instance
_router = SemanticRouter()


def route(transcript: str) -> tuple[str, dict | None]:
    return _router.route(transcript)


def get_installed_apps() -> dict[str, str]:
    try:
        installed = set(os.listdir("/Applications"))
        home_apps = os.path.expanduser("~/Applications")
        if os.path.isdir(home_apps):
            installed.update(os.listdir(home_apps))
    except Exception:
        installed = set()
    registry = {}
    for category, apps in APP_CATEGORIES.items():
        for app in apps:
            if f"{app}.app" in installed or app in installed:
                registry[category] = app
                break
    return registry


def get_mechanism(app_name: str) -> list[str]:
    return APP_MECHANISMS.get(app_name, ["accessibility"])


def build_app_context() -> str:
    apps = get_installed_apps()
    if not apps:
        return ""
    lines = ["INSTALLED APPS:"]
    for cat, app in sorted(apps.items()):
        mechs = get_mechanism(app)
        lines.append(f"  {cat}: {app} (via {', '.join(mechs)})")
    return "\n".join(lines)
