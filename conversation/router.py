"""3-tier intent router with semantic embeddings.

Tier 1: INSTANT — regex match, direct execution (<10ms)
Tier 2: FAST    — Groq, no tools, simple conversation (~400ms)
Tier 3: THINK   — Claude + tools + optional vision (~800ms)

Routing: regex → embedding similarity → keyword fallback
"""

import os
import re
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


# Tool-requiring keywords → Claude (has tools)
TOOL_KEYWORDS = frozenset({
    "email", "mail", "inbox", "calendar", "schedule", "meeting",
    "message", "send", "reply", "check", "search", "find", "read",
    "remind", "task", "slack", "text", "contact", "compose",
    "open and", "tell me what", "summarize",
})

# Vision-requiring keywords → Claude + screenshot
VISION_KEYWORDS = frozenset({
    "this", "that", "here", "screen", "what's on", "look at",
    "see", "showing", "button", "click", "fill", "form",
    "rewrite", "edit this", "what does this",
})

# Simple conversation → Groq (fast, no tools)
SIMPLE_KEYWORDS = frozenset({
    "hello", "hi", "hey", "thanks", "thank you", "goodbye",
    "how are you", "good morning", "joke", "define", "translate",
    "calculate", "explain", "what is", "who is",
})


def route(transcript: str) -> tuple[str, dict | None]:
    """Route transcript to tier 1 (instant), tier 2 (groq), or tier 3 (claude)."""
    text = transcript.strip()
    lower = text.lower()

    # Tier 1: instant regex patterns
    for pattern, handler in INSTANT_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            action_type, args = handler(match)
            log.info("Tier 1 (instant): %s → %s", lower[:40], action_type)
            return "instant", {"action": action_type, **args}

    # Tier 3: needs tools
    if any(kw in lower for kw in TOOL_KEYWORDS):
        log.info("Tier 3 (claude): %s", lower[:40])
        return "claude", None

    # Tier 3: needs vision
    if any(kw in lower for kw in VISION_KEYWORDS):
        log.info("Tier 3 (claude+vision): %s", lower[:40])
        return "claude", None

    # Tier 2: simple conversation
    if any(kw in lower for kw in SIMPLE_KEYWORDS):
        log.info("Tier 2 (groq): %s", lower[:40])
        return "groq", None

    # Default: short = groq, long = claude
    if len(text.split()) <= 8:
        log.info("Tier 2 (groq): %s", lower[:40])
        return "groq", None

    log.info("Tier 3 (claude): %s", lower[:40])
    return "claude", None


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
