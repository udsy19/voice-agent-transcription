"""Quick capture — route voice to the right place automatically.

"remind me to X" → todo
"note: X" / "remember that X" → brain memory
"schedule X" / "book X" → assistant (calendar)
"""

import re
from logger import get_logger

log = get_logger("capture")

# Patterns ordered by priority
_PATTERNS = [
    # Todos
    (r'^(?:remind me to|add (?:a )?todo|todo:?|task:?)\s+(.+)', "todo"),
    (r'^(?:i need to|don\'t forget to|make sure to)\s+(.+)', "todo"),

    # Memory/notes
    (r'^(?:note:?|remember (?:that|this):?|important:?)\s+(.+)', "memory"),
    (r'^(?:keep in mind|for the record)\s+(.+)', "memory"),

    # Meeting notes
    (r'^(?:meeting notes?:?|from (?:the|my) meeting:?)\s+(.+)', "meeting_notes"),
    (r'^(?:action items?:?|takeaways?:?)\s+(.+)', "meeting_notes"),

    # Calendar (route to assistant)
    (r'^(?:schedule|book|add (?:a |an )?(?:event|meeting|call|appointment))\s+(.+)', "calendar"),
]


def detect(raw_text: str) -> tuple[str, str] | None:
    """Detect if text is a quick capture command.

    Returns (intent, extracted_text) or None to fall through to normal dictation.
    intent is one of: "todo", "memory", "meeting_notes", "calendar"
    """
    text = raw_text.strip()
    if len(text) < 5:
        return None

    text_lower = text.lower()
    for pattern, intent in _PATTERNS:
        m = re.match(pattern, text_lower)
        if m:
            # Use original case for the captured text
            extracted = text[m.start(1):m.end(1)].strip()
            if extracted:
                log.info("Quick capture: %s → %s", intent, extracted[:50])
                return (intent, extracted)

    return None
