"""Morning briefing — compose a spoken summary of your day."""

import time
from logger import get_logger

log = get_logger("briefing")


def compose(calendar_events: list, todos: list, deadlines: list, facts: list) -> str:
    """Compose a natural morning briefing.

    Returns a string meant to be spoken aloud via TTS.
    """
    parts = []
    now = time.strftime("%A, %B %d")

    parts.append(f"Good morning. Here's your {now} briefing.")

    # Calendar
    if calendar_events:
        n = len(calendar_events)
        if n == 1:
            ev = calendar_events[0]
            parts.append(f"You have one event today: {ev['summary']}.")
        else:
            parts.append(f"You have {n} events today.")
            for ev in calendar_events[:5]:
                # Extract time from ISO string
                start = ev.get("start", "")
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    t = dt.strftime("%-I:%M %p")
                except Exception:
                    t = start
                parts.append(f"{ev['summary']} at {t}.")
    else:
        parts.append("Your calendar is clear today.")

    # Todos
    if todos:
        n = len(todos)
        if n <= 3:
            items = ", ".join(t["text"] for t in todos)
            parts.append(f"You have {n} pending task{'s' if n > 1 else ''}: {items}.")
        else:
            top = ", ".join(t["text"] for t in todos[:3])
            parts.append(f"You have {n} pending tasks. Top ones: {top}.")

    # Deadlines
    if deadlines:
        for d in deadlines[:2]:
            parts.append(f"Reminder: {d['text']}.")

    if not calendar_events and not todos and not deadlines:
        parts.append("Nothing urgent — looks like a free day.")

    return " ".join(parts)
