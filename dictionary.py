import json
import os
import re
import threading
from logger import get_logger

log = get_logger("dictionary")

DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_dictionary.json")
_file_lock = threading.Lock()


class PersonalDictionary:
    def __init__(self):
        self._corrections: dict[str, str] = {}
        self._terms: list[str] = []
        self._load()

    def _load(self):
        if not os.path.exists(DICT_PATH):
            return
        try:
            with open(DICT_PATH) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._corrections = data.get("corrections") or {}
                self._terms = data.get("terms") or []
                # Validate types
                if not isinstance(self._corrections, dict):
                    self._corrections = {}
                if not isinstance(self._terms, list):
                    self._terms = []
            log.info("Loaded %d corrections, %d terms", len(self._corrections), len(self._terms))
        except (json.JSONDecodeError, IOError) as e:
            log.error("Failed to load dictionary: %s, starting fresh", e)
            self._corrections = {}
            self._terms = []

    def _save(self):
        with _file_lock:
            try:
                with open(DICT_PATH, "w") as f:
                    json.dump({"corrections": self._corrections, "terms": self._terms}, f, indent=2)
            except IOError as e:
                log.error("Failed to save dictionary: %s", e)

    def add_correction(self, wrong: str, correct: str):
        self._corrections[wrong.lower()] = correct
        if correct not in self._terms:
            self._terms.append(correct)
        self._save()
        log.info("Learned: '%s' -> '%s'", wrong, correct)

    def add_term(self, term: str):
        if term not in self._terms:
            self._terms.append(term)
            self._save()
            log.info("Added term: '%s'", term)

    def remove_term(self, term: str):
        if term in self._terms:
            self._terms.remove(term)
            self._save()
            log.info("Removed term: '%s'", term)

    def apply(self, text: str) -> str:
        for wrong, correct in list(self._corrections.items()):
            try:
                text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
            except re.error:
                pass
        return text

    @property
    def terms(self) -> list[str]:
        return self._terms.copy()

    @property
    def corrections(self) -> dict[str, str]:
        return self._corrections.copy()

    def get_whisper_prompt(self) -> str | None:
        if not self._terms:
            return None
        return ", ".join(self._terms[:30])

    def list_all(self) -> str:
        lines = []
        if self._terms:
            lines.append("Terms:")
            for t in self._terms:
                lines.append(f"  - {t}")
        if self._corrections:
            lines.append("Corrections:")
            for wrong, correct in self._corrections.items():
                lines.append(f"  - '{wrong}' -> '{correct}'")
        return "\n".join(lines) if lines else "(empty)"
