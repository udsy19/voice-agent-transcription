"""Mem0-powered memory — learns about you from every conversation.

Automatically extracts facts, preferences, people, and context.
Stored locally in Qdrant vector DB. Searchable by the assistant.
"""

import os
import time
import threading
from logger import get_logger
from config import DATA_DIR, GROQ_API_KEY

log = get_logger("memory")

MEMORY_DB_PATH = str(DATA_DIR / "memory_db")
_mem = None
_lock = threading.Lock()


def _get_mem0():
    """Lazy-load mem0 instance."""
    global _mem
    if _mem is not None:
        return _mem
    with _lock:
        if _mem is not None:
            return _mem
        try:
            from mem0 import Memory
            config = {
                "llm": {
                    "provider": "groq",
                    "config": {
                        "model": "llama-3.1-8b-instant",
                        "api_key": GROQ_API_KEY,
                    }
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "embedding_dims": 384,
                    }
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "muse",
                        "embedding_model_dims": 384,
                        "path": MEMORY_DB_PATH,
                    }
                },
            }
            _mem = Memory.from_config(config)
            log.info("Mem0 memory loaded (db: %s)", MEMORY_DB_PATH)
            return _mem
        except Exception as e:
            log.error("Failed to load mem0: %s", e)
            return None


def remember(text: str, user_id: str = "user", metadata: dict = None):
    """Store a memory. Mem0 auto-extracts facts."""
    m = _get_mem0()
    if not m:
        return []
    try:
        result = m.add(text, user_id=user_id, metadata=metadata or {})
        entries = result.get("results", [])
        log.info("Remembered %d facts from: %s", len(entries), text[:60])
        return entries
    except Exception as e:
        log.error("Memory add failed: %s", e)
        return []


def remember_conversation(messages: list[dict], user_id: str = "user"):
    """Store memories from a conversation (list of {role, content} dicts)."""
    m = _get_mem0()
    if not m:
        return []
    try:
        result = m.add(messages, user_id=user_id)
        entries = result.get("results", [])
        log.info("Learned %d facts from conversation", len(entries))
        return entries
    except Exception as e:
        log.error("Conversation memory failed: %s", e)
        return []


def recall(query: str, user_id: str = "user", limit: int = 10) -> list[dict]:
    """Search memories. Returns list of {memory, score, metadata}."""
    m = _get_mem0()
    if not m:
        return []
    try:
        results = m.search(query, user_id=user_id, limit=limit)
        return results.get("results", [])
    except Exception as e:
        log.error("Memory search failed: %s", e)
        return []


def get_all(user_id: str = "user") -> list[dict]:
    """Get all stored memories."""
    m = _get_mem0()
    if not m:
        return []
    try:
        results = m.get_all(user_id=user_id)
        return results.get("results", [])
    except Exception as e:
        log.error("Memory get_all failed: %s", e)
        return []


def delete(memory_id: str):
    """Delete a specific memory."""
    m = _get_mem0()
    if not m:
        return
    try:
        m.delete(memory_id)
        log.info("Deleted memory: %s", memory_id)
    except Exception as e:
        log.error("Memory delete failed: %s", e)


def get_context_for_llm(query: str, user_id: str = "user") -> str:
    """Get relevant memories as context string for the assistant prompt."""
    memories = recall(query, user_id=user_id, limit=8)
    if not memories:
        return ""
    lines = [f"- {m['memory']}" for m in memories if m.get("memory")]
    if not lines:
        return ""
    return "\n\nRelevant memories about the user:\n" + "\n".join(lines)


def remember_async(text: str, user_id: str = "user", metadata: dict = None):
    """Store memory in background thread (non-blocking)."""
    threading.Thread(
        target=remember, args=(text, user_id, metadata),
        daemon=True,
    ).start()
