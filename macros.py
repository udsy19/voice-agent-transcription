"""Voice macros — chainable voice-triggered workflows.

A macro can:
  - Set a tone/style
  - Insert template text
  - Activate a domain mode
  - Run a shell command

Triggered by a spoken phrase like "email mode" or "standup notes".
"""

import json
import os
from logger import get_logger

log = get_logger("macros")

MACROS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macros.json")

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
        if os.path.exists(MACROS_PATH):
            try:
                with open(MACROS_PATH) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._macros.update(data)
                log.info("Loaded %d macros", len(self._macros))
            except (json.JSONDecodeError, IOError) as e:
                log.error("Failed to load macros: %s", e)

    def _save(self):
        # Only save custom macros (not builtins)
        custom = {k: v for k, v in self._macros.items() if k not in BUILTIN_MACROS}
        try:
            with open(MACROS_PATH, "w") as f:
                json.dump(custom, f, indent=2)
        except IOError as e:
            log.error("Failed to save macros: %s", e)

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
        """
        import time
        results = []
        for action in macro.get("actions", []):
            atype = action.get("type", "")
            value = action.get("value", "")

            # Template substitution
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
            # "shell" action type deliberately removed for security

        log.info("Executed macro: %d actions", len(results))
        return results
