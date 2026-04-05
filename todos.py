"""Todo list — persistent, simple, voice-friendly."""

import time
import uuid
import threading
import safe_json
from config import DATA_DIR
from logger import get_logger

log = get_logger("todos")

TODOS_PATH = str(DATA_DIR / "todos.json")


class TodoList:
    def __init__(self):
        data = safe_json.load(TODOS_PATH, {"items": []})
        self._items: list[dict] = data.get("items", [])
        self._lock = threading.Lock()

    def _save(self):
        with self._lock:
            safe_json.save(TODOS_PATH, {"items": self._items})

    def add(self, text: str) -> dict:
        item = {
            "id": uuid.uuid4().hex[:8],
            "text": text.strip(),
            "done": False,
            "created": time.time(),
        }
        self._items.append(item)
        self._save()
        log.info("Added todo: %s", text[:50])
        return item

    def complete(self, todo_id: str) -> bool:
        for item in self._items:
            if item["id"] == todo_id:
                item["done"] = True
                item["completed_at"] = time.time()
                self._save()
                log.info("Completed: %s", item["text"][:50])
                return True
        return False

    def remove(self, todo_id: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if i["id"] != todo_id]
        if len(self._items) < before:
            self._save()
            return True
        return False

    def list_all(self) -> list[dict]:
        return self._items.copy()

    def list_pending(self) -> list[dict]:
        return [i for i in self._items if not i["done"]]

    def list_done(self) -> list[dict]:
        return [i for i in self._items if i["done"]]

    def summary_for_llm(self) -> str:
        pending = self.list_pending()
        if not pending:
            return ""
        lines = [f"- {i['text']}" for i in pending[:10]]
        return f"\nPending todos ({len(pending)}):\n" + "\n".join(lines)
