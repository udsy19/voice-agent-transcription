import os
import threading
from difflib import SequenceMatcher
import safe_json
from logger import get_logger

log = get_logger("snippets")

from config import DATA_DIR
SNIPPETS_PATH = str(DATA_DIR / "snippets.json")
SHARED_SNIPPETS_PATH = str(DATA_DIR / "shared_snippets.json")
MATCH_THRESHOLD = 0.75
_file_lock = threading.Lock()


class SnippetStore:
    def __init__(self):
        self._snippets: dict[str, dict] = {}
        self._shared: dict[str, dict] = {}
        self._load()

    def _load(self):
        for path, attr in [(SNIPPETS_PATH, "_snippets"), (SHARED_SNIPPETS_PATH, "_shared")]:
            data = safe_json.load(path, {})
            if isinstance(data, dict):
                setattr(self, attr, data)
                log.info("Loaded %d %ssnippets", len(data), "shared " if attr == "_shared" else "")

    def _save(self):
        with _file_lock:
            safe_json.save(SNIPPETS_PATH, self._snippets)

    def add(self, trigger: str, text: str, description: str = ""):
        self._snippets[trigger.lower()] = {"text": text, "description": description}
        self._save()
        log.info("Added snippet: '%s'", trigger)

    def remove(self, trigger: str):
        key = trigger.lower()
        if key in self._snippets:
            del self._snippets[key]
            self._save()
            log.info("Removed snippet: '%s'", trigger)

    def match(self, spoken_text: str) -> str | None:
        spoken_lower = spoken_text.lower().strip()
        # Snapshot to avoid mutation during iteration
        all_snippets = {**self._shared, **self._snippets}

        if spoken_lower in all_snippets:
            return all_snippets[spoken_lower].get("text", "")

        for prefix in ["insert ", "snippet ", "paste "]:
            stripped = spoken_lower.removeprefix(prefix)
            if stripped != spoken_lower and stripped in all_snippets:
                return all_snippets[stripped].get("text", "")

        best_match = None
        best_ratio = 0.0
        for trigger, data in all_snippets.items():
            try:
                ratio = SequenceMatcher(None, spoken_lower, trigger).ratio()
            except Exception:
                continue
            if ratio > best_ratio and ratio >= MATCH_THRESHOLD:
                best_ratio = ratio
                best_match = data.get("text", "")

        if best_match:
            log.info("Fuzzy matched snippet (%.0f%%)", best_ratio * 100)
            return best_match
        return None

    def list_all(self) -> str:
        lines = []
        if self._snippets:
            lines.append("Personal Snippets:")
            for trigger, data in self._snippets.items():
                preview = data.get("text", "")[:60].replace("\n", " ")
                lines.append(f"  '{trigger}' -> {preview}{'...' if len(data.get('text', '')) > 60 else ''}")
        if self._shared:
            lines.append("Shared Snippets:")
            for trigger, data in self._shared.items():
                preview = data.get("text", "")[:60].replace("\n", " ")
                lines.append(f"  '{trigger}' -> {preview}")
        return "\n".join(lines) if lines else "(no snippets)"
