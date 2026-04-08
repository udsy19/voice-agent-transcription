"""Brain — long-term memory, deadlines, meeting notes, habit detection."""

import time
import re
import threading
import safe_json
from config import DATA_DIR
from logger import get_logger

log = get_logger("brain")

BRAIN_PATH = str(DATA_DIR / "brain.json")


class Brain:
    def __init__(self):
        self._lock = threading.Lock()
        data = safe_json.load(BRAIN_PATH, {"facts": [], "deadlines": [], "meetings": [], "habits": {}})
        self.facts: list[dict] = data.get("facts", [])
        self.deadlines: list[dict] = data.get("deadlines", [])
        self.meetings: list[dict] = data.get("meetings", [])
        self.habits: dict = data.get("habits", {})

    def _save(self):
        with self._lock:
            safe_json.save(BRAIN_PATH, {
                "facts": self.facts[-200:],
                "deadlines": self.deadlines[-50:],
                "meetings": self.meetings[-50:],
                "habits": self.habits,
            })

    def remember(self, text: str, category: str = "general"):
        """Store a fact/note in long-term memory."""
        self.facts.append({"text": text.strip(), "category": category, "ts": time.time()})
        self._save()
        log.info("Remembered: %s", text[:60])

    def add_deadline(self, text: str, due: str):
        """Track a deadline."""
        self.deadlines.append({"text": text, "due": due, "created": time.time(), "reminded": False})
        self._save()
        log.info("Deadline: %s by %s", text[:40], due)

    def get_deadlines(self, upcoming_days: int = 7) -> list[dict]:
        """Get upcoming, non-reminded deadlines."""
        results = []
        for d in self.deadlines:
            if d.get("reminded"):
                continue
            # Try to parse the due date; fall back to including it
            due = d.get("due", "")
            if due:
                try:
                    from datetime import datetime, timedelta
                    # Try common date formats
                    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
                        try:
                            due_dt = datetime.strptime(due, fmt)
                            # Include if due date is in the future or within the window
                            if due_dt <= datetime.now() + timedelta(days=upcoming_days):
                                results.append(d)
                            break
                        except ValueError:
                            continue
                    else:
                        # Unparseable due date — include it anyway (user set it)
                        results.append(d)
                except Exception:
                    results.append(d)
            else:
                # No due date — include if created recently
                results.append(d)
        return results

    def add_meeting_notes(self, summary: str, date: str, notes: str, action_items: list[str]):
        """Store meeting notes + action items."""
        self.meetings.append({
            "summary": summary, "date": date, "notes": notes,
            "action_items": action_items, "ts": time.time(),
        })
        self._save()
        log.info("Meeting notes: %s (%d actions)", summary[:40], len(action_items))

    def get_meeting_context(self, query: str) -> str:
        """Search past meetings for context (name, topic, etc.)."""
        query_lower = query.lower()
        matches = []
        for m in reversed(self.meetings):
            if (query_lower in m["summary"].lower() or
                query_lower in m.get("notes", "").lower()):
                matches.append(m)
            if len(matches) >= 3:
                break
        if not matches:
            return ""
        lines = []
        for m in matches:
            lines.append(f"Meeting: {m['summary']} ({m['date']})")
            if m.get("notes"):
                lines.append(f"  Notes: {m['notes'][:150]}")
            if m.get("action_items"):
                lines.append(f"  Actions: {', '.join(m['action_items'][:5])}")
        return "\n".join(lines)

    def detect_deadline_in_text(self, text: str) -> dict | None:
        """Extract deadline from natural text like 'finish X by Friday'."""
        patterns = [
            r'(?:due|by|before|deadline[:\s]*)\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next week|end of (?:day|week|month))',
            r'(?:need to|have to|must)\s+(.{5,40}?)\s+(?:by|before)\s+(\w+)',
            r'(?:finish|complete|submit|send)\s+(.{5,40}?)\s+(?:by|before)\s+(\w+)',
        ]
        text_lower = text.lower()
        for pattern in patterns:
            m = re.search(pattern, text_lower)
            if m:
                return {"text": text.strip(), "due_hint": m.group(0)}
        return None

    def get_context_for_llm(self) -> str:
        """Get brain context to inject into assistant system prompt."""
        parts = []

        # Recent facts (last 10)
        recent_facts = self.facts[-10:]
        if recent_facts:
            parts.append("Things I remember:")
            for f in recent_facts:
                parts.append(f"- {f['text']}")

        # Upcoming deadlines
        deadlines = self.get_deadlines()
        if deadlines:
            parts.append("\nUpcoming deadlines:")
            for d in deadlines:
                parts.append(f"- {d['text']} (due: {d.get('due', 'soon')})")

        # Recent meeting context
        recent_meetings = self.meetings[-3:]
        if recent_meetings:
            parts.append("\nRecent meetings:")
            for m in recent_meetings:
                actions = ", ".join(m.get("action_items", [])[:3])
                parts.append(f"- {m['summary']}: {actions}" if actions else f"- {m['summary']}")

        return "\n".join(parts) if parts else ""
