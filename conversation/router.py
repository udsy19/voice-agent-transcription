"""3-tier intent router — decides which execution path BEFORE any LLM call.

Tier 1: INSTANT — regex match, direct execution, no LLM (<100ms)
Tier 2: FAST    — Groq + tools, no vision (~400ms)
Tier 3: THINK   — Claude + vision + agentic loop (~800ms+)
"""

import re
import os
from logger import get_logger

log = get_logger("router")

# ── Tier 1: Instant patterns (no LLM) ───────────────────────────────────────

INSTANT_PATTERNS = [
    # Open apps
    (r"\b(?:open|launch|start|switch to)\s+(\w[\w\s]*?)(?:\s+app)?$",
     lambda m: ("open_app", {"app": m.group(1).strip()})),

    # Media control
    (r"\b(?:play|pause|resume)\s*(?:music|song|track)?$",
     lambda _: ("media", {"action": "playpause"})),
    (r"\b(?:next|skip)\s*(?:song|track)?$",
     lambda _: ("media", {"action": "next"})),
    (r"\b(?:previous)\s*(?:song|track)?$",
     lambda _: ("media", {"action": "previous"})),

    # Volume
    (r"\bvolume\s*(up|down)$",
     lambda m: ("volume", {"direction": m.group(1)})),
    (r"\b(?:mute|unmute)$",
     lambda m: ("volume", {"direction": "mute"})),

    # Screenshot
    (r"\btake\s*(?:a\s+)?screenshot$",
     lambda _: ("screenshot", {})),

    # Type/dictate
    (r"\btype\s+(.+)$",
     lambda m: ("type", {"text": m.group(1)})),

    # Undo
    (r"\bundo\s*(?:that)?$",
     lambda _: ("undo", {})),

    # Lock/sleep
    (r"\block\s*(?:the\s+)?(?:screen|computer)$",
     lambda _: ("lock", {})),
]

# ── Tier 2: Groq-level keywords (fast LLM, no vision) ───────────────────────

GROQ_KEYWORDS = {
    "email", "mail", "inbox", "calendar", "schedule", "meeting",
    "search", "find", "look up", "look for", "what time", "when is",
    "remind", "reminder", "task", "todo", "slack", "message", "send",
    "weather", "news", "price", "stock", "translate", "define",
    "set timer", "set alarm", "calculate", "how much", "how many",
    "who is", "what is", "tell me", "summarize", "read",
    "reply", "respond", "forward", "compose", "draft",
}

# ── Tier 3: Vision keywords (need Claude + screenshot) ───────────────────────

VISION_KEYWORDS = {
    "this", "that", "here", "screen", "what's on", "what is on",
    "look at", "see", "showing", "displayed", "button", "click",
    "fill out", "form", "page", "tab", "window", "rewrite",
    "edit this", "change this", "fix this", "what does this",
}


def route(transcript: str) -> tuple[str, dict | None]:
    """Route a transcript to the appropriate tier.

    Returns:
        ("instant", {"action": ..., "args": ...})
        ("groq", None)
        ("claude", None)
    """
    text = transcript.strip()
    lower = text.lower()

    # Tier 1: instant patterns
    for pattern, handler in INSTANT_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            action_type, args = handler(match)
            log.info("Tier 1 (instant): %s → %s", lower[:40], action_type)
            return "instant", {"action": action_type, **args}

    # Tier 3: vision keywords (check before Groq — more specific)
    if any(kw in lower for kw in VISION_KEYWORDS):
        log.info("Tier 3 (claude): %s", lower[:40])
        return "claude", None

    # Tier 2: Groq keywords
    if any(kw in lower for kw in GROQ_KEYWORDS):
        log.info("Tier 2 (groq): %s", lower[:40])
        return "groq", None

    # Default: short = groq, long/complex = claude
    word_count = len(text.split())
    if word_count <= 10:
        log.info("Tier 2 (groq, short): %s", lower[:40])
        return "groq", None
    else:
        log.info("Tier 3 (claude, complex): %s", lower[:40])
        return "claude", None


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

# Which mechanism each app supports (priority order)
APP_MECHANISMS = {
    "Mail": ["applescript"],
    "Calendar": ["applescript"],
    "Contacts": ["applescript"],
    "Finder": ["applescript"],
    "Safari": ["applescript", "accessibility"],
    "Messages": ["applescript"],
    "Notes": ["applescript"],
    "Reminders": ["applescript"],
    "Music": ["applescript"],
    "Terminal": ["applescript"],
    "Spark": ["accessibility"],
    "Slack": ["accessibility", "applescript"],
    "Chrome": ["applescript", "accessibility"],
    "Arc": ["accessibility"],
    "Notion": ["accessibility"],
    "Linear": ["accessibility"],
    "VS Code": ["accessibility"],
    "Cursor": ["accessibility"],
    "Figma": ["accessibility"],
}


def get_installed_apps() -> dict[str, str]:
    """Scan /Applications and return {category: preferred_app}."""
    try:
        installed = set(os.listdir("/Applications"))
        # Also check ~/Applications
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
    """Get the preferred interaction mechanisms for an app."""
    return APP_MECHANISMS.get(app_name, ["accessibility"])


def build_app_context() -> str:
    """Build an app registry context string for the system prompt."""
    apps = get_installed_apps()
    if not apps:
        return ""
    lines = ["INSTALLED APPS:"]
    for cat, app in sorted(apps.items()):
        mechs = get_mechanism(app)
        lines.append(f"  {cat}: {app} (via {', '.join(mechs)})")
    return "\n".join(lines)
