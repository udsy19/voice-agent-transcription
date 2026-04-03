import os
import re
import threading
import safe_json
from logger import get_logger

log = get_logger("dictionary")

from config import DATA_DIR
DICT_PATH = str(DATA_DIR / "personal_dictionary.json")
_file_lock = threading.Lock()


class PersonalDictionary:
    def __init__(self):
        self._corrections: dict[str, str] = {}
        self._terms: list[str] = []
        self._load()

    def _load(self):
        data = safe_json.load(DICT_PATH, {"corrections": {}, "terms": []})
        if isinstance(data, dict):
            self._corrections = data.get("corrections") or {}
            self._terms = data.get("terms") or []
            if not isinstance(self._corrections, dict):
                self._corrections = {}
            if not isinstance(self._terms, list):
                self._terms = []
        log.info("Loaded %d corrections, %d terms", len(self._corrections), len(self._terms))

    def _save(self):
        with _file_lock:
            safe_json.save(DICT_PATH, {"corrections": self._corrections, "terms": self._terms})

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
                # Use lambda to prevent backreference interpretation in replacement
                text = re.sub(re.escape(wrong), lambda m: correct, text, flags=re.IGNORECASE)
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
