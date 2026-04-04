"""Weekly reflection — compile your week's data into a spoken summary."""

import time
from datetime import datetime, timedelta
from logger import get_logger

log = get_logger("reflection")


def compose_week(brain=None, todos=None, oauth=None, memories=None) -> str:
    """Compile a weekly reflection summary."""
    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    week_ts = time.time() - 7 * 86400

    parts = []
    parts.append(f"Here's your week in review, {week_start.strftime('%B %d')} to {now.strftime('%B %d')}.")

    # Meetings this week
    if brain and brain.meetings:
        recent = [m for m in brain.meetings if m.get("ts", 0) > week_ts]
        if recent:
            parts.append(f"You had {len(recent)} meeting{'s' if len(recent) != 1 else ''} this week.")
            for m in recent[:3]:
                parts.append(f"{m['summary']}.")

    # Calendar events this week
    if oauth:
        try:
            token = oauth.get_token("google")
            if token:
                from integrations.google_calendar import list_events
                result = list_events(token, days_ahead=1, max_results=20)
                if result.get("ok"):
                    parts.append(f"You had {len(result['events'])} calendar events.")
        except Exception:
            pass

    # Todos completed
    if todos:
        done = todos.list_done()
        recent_done = [t for t in done if t.get("completed_at", 0) > week_ts]
        pending = todos.list_pending()
        if recent_done:
            parts.append(f"You completed {len(recent_done)} task{'s' if len(recent_done) != 1 else ''}.")
        if pending:
            parts.append(f"You have {len(pending)} task{'s' if len(pending) != 1 else ''} still pending.")

    # Deadlines
    if brain:
        deadlines = brain.get_deadlines(upcoming_days=7)
        if deadlines:
            parts.append(f"Upcoming deadline{'s' if len(deadlines) != 1 else ''}: {', '.join(d['text'][:40] for d in deadlines[:3])}.")

    # Key memories/learnings
    if brain and brain.facts:
        recent_facts = [f for f in brain.facts if f.get("ts", 0) > week_ts]
        if recent_facts:
            parts.append(f"You noted {len(recent_facts)} thing{'s' if len(recent_facts) != 1 else ''} this week.")

    if len(parts) <= 1:
        parts.append("Looks like a quiet week — no meetings, todos, or notes recorded.")

    return " ".join(parts)
