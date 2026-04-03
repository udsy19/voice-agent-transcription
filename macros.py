"""Voice macros — chainable voice-triggered workflows.

A macro can:
  - Set a tone/style
  - Insert template text
  - Activate a domain mode
  - Run a shell command

Triggered by a spoken phrase like "email mode" or "standup notes".
"""

import os
import safe_json
from logger import get_logger

log = get_logger("macros")

from config import DATA_DIR
MACROS_PATH = str(DATA_DIR / "macros.json")

BUILTIN_MACROS = {
    "email mode": {
        "description": "Formal email drafting",
        "actions": [
            {"type": "set_tone", "value": "formal"},
            {"type": "insert_text", "value": "Dear "},
        ],
    },
    "slack mode": {
        "description": "Casual messaging",
        "actions": [
            {"type": "set_tone", "value": "casual"},
        ],
    },
    "code mode": {
        "description": "Code dictation",
        "actions": [
            {"type": "set_tone", "value": "code"},
            {"type": "set_domain", "value": "tech"},
        ],
    },
    "standup notes": {
        "description": "Daily standup format",
        "actions": [
            {"type": "insert_text", "value": "Yesterday:\n- \n\nToday:\n- \n\nBlockers:\n- "},
        ],
    },
    "meeting notes": {
        "description": "Meeting notes template",
        "actions": [
            {"type": "insert_text", "value": "Meeting Notes\nDate: {date}\nAttendees: \n\nAgenda:\n1. \n\nAction Items:\n- "},
        ],
    },
}


class MacroEngine:
    def __init__(self):
        self._macros: dict = {}
        self._load()

    def _load(self):
        self._macros = BUILTIN_MACROS.copy()
        data = safe_json.load(MACROS_PATH, {})
        if isinstance(data, dict):
            self._macros.update(data)
        log.info("Loaded %d macros", len(self._macros))

    def _save(self):
        custom = {k: v for k, v in self._macros.items() if k not in BUILTIN_MACROS}
        safe_json.save(MACROS_PATH, custom)

    def match(self, spoken_text: str) -> dict | None:
        """Check if spoken text triggers a macro. Returns macro dict or None."""
        spoken_lower = spoken_text.lower().strip()
        # Exact match
        if spoken_lower in self._macros:
            return self._macros[spoken_lower]
        # Prefix match ("activate email mode" → "email mode")
        for trigger, macro in self._macros.items():
            if spoken_lower.endswith(trigger):
                return macro
        return None

    def add(self, trigger: str, description: str, actions: list[dict]):
        self._macros[trigger.lower()] = {
            "description": description,
            "actions": actions,
        }
        self._save()
        log.info("Added macro: '%s'", trigger)

    def remove(self, trigger: str):
        key = trigger.lower()
        if key in self._macros:
            del self._macros[key]
            self._save()
            log.info("Removed macro: '%s'", trigger)

    def list_all(self) -> dict:
        return {k: v.get("description", "") for k, v in self._macros.items()}

    def execute(self, macro: dict, context: dict) -> list[dict]:
        """Execute macro actions. Returns list of results.

        context should contain:
          - set_tone: callable(tone_name)
          - set_domain: callable(domain_name)
          - inject_text: callable(text)
          - get_app: callable() -> str  (current app name)

        Supported action types:
          - set_tone: Set the cleaning tone
          - set_domain: Activate a domain
          - insert_text: Paste template text (supports {date}, {time})
          - delay: Pause for N seconds (max 5)
          - condition: Only continue if current app matches value
          - repeat: Repeat the next N actions a given number of times
        """
        import time
        results = []
        actions = macro.get("actions", [])
        i = 0
        while i < len(actions):
            action = actions[i]
            atype = action.get("type", "")
            value = action.get("value", "")

            # Template substitution
            if isinstance(value, str):
                if "{date}" in value:
                    value = value.replace("{date}", time.strftime("%Y-%m-%d"))
                if "{time}" in value:
                    value = value.replace("{time}", time.strftime("%H:%M"))

            if atype == "set_tone" and "set_tone" in context:
                context["set_tone"](value)
                results.append({"type": "set_tone", "value": value})

            elif atype == "set_domain" and "set_domain" in context:
                context["set_domain"](value)
                results.append({"type": "set_domain", "value": value})

            elif atype == "insert_text" and "inject_text" in context:
                context["inject_text"](value)
                results.append({"type": "insert_text", "chars": len(value)})

            elif atype == "delay":
                # Pause between actions (capped at 5s to prevent abuse)
                try:
                    secs = min(float(value), 5.0)
                except (ValueError, TypeError):
                    secs = 0.5
                time.sleep(secs)
                results.append({"type": "delay", "seconds": secs})

            elif atype == "condition":
                # Only continue if current app matches value
                get_app = context.get("get_app")
                current_app = get_app() if get_app else ""
                if value.lower() not in current_app.lower():
                    log.info("Condition failed: '%s' not in '%s', stopping macro", value, current_app)
                    results.append({"type": "condition", "matched": False, "app": current_app})
                    break
                results.append({"type": "condition", "matched": True, "app": current_app})

            elif atype == "repeat":
                # Repeat the next N actions a given number of times
                try:
                    count = max(1, min(int(action.get("count", 2)), 10))
                    n_actions = max(1, min(int(action.get("n", 1)), 5))
                except (ValueError, TypeError):
                    count, n_actions = 2, 1
                repeat_actions = actions[i + 1: i + 1 + n_actions]
                if repeat_actions:
                    for _ in range(count - 1):  # -1 because they'll run once normally
                        actions = actions[:i + 1 + n_actions] + repeat_actions + actions[i + 1 + n_actions:]
                results.append({"type": "repeat", "count": count, "actions": n_actions})

            # "shell" action type deliberately removed for security

            i += 1

        log.info("Executed macro: %d actions", len(results))
        return results
