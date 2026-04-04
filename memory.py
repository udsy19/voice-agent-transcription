"""Mem0-powered memory — learns about you from every conversation.

Features:
- Auto-extracts facts from conversations and dictations
- Deduplication: won't store the same fact twice
- Growth cap: prunes oldest memories when over limit
- Rate limiting: throttles Groq calls to avoid limits
- Offline resilience: caches and retries on network failure
- Category tagging via LLM extraction
- Conflict resolution: new facts update old ones
- Export/import for backup
- Provides context for both assistant and dictation
"""

import os
import time
import json
import threading
from logger import get_logger
from config import DATA_DIR, GROQ_API_KEY
import safe_json

log = get_logger("memory")

MEMORY_DB_PATH = str(DATA_DIR / "memory_db")
MAX_MEMORIES = 500  # cap total memories
RATE_LIMIT_INTERVAL = 15.0  # min seconds between mem0 writes (Groq free = 6K TPM)
EXPORT_PATH = str(DATA_DIR / "memories_export.json")

_mem = None
_lock = threading.Lock()
_last_write_ts = 0.0
_pending_queue: list[str] = []  # queue for offline/rate-limited writes
_queue_lock = threading.Lock()


def _get_mem0():
    """Lazy-load mem0 instance."""
    global _mem
    if _mem is not None:
        return _mem
    with _lock:
        if _mem is not None:
            return _mem
        if not GROQ_API_KEY:
            log.warning("No Groq key — memory disabled")
            return None
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
            log.info("Mem0 loaded (%s)", MEMORY_DB_PATH)
            return _mem
        except Exception as e:
            log.error("Mem0 init failed: %s", e)
            return None


# ── Core operations ─────────────────────────────────────────────────────────

def remember(text: str, user_id: str = "user", metadata: dict = None) -> list:
    """Store a memory with rate limiting and deduplication (handled by mem0)."""
    global _last_write_ts
    if not text or len(text.strip()) < 5:
        return []

    # Rate limit: don't hammer Groq
    now = time.time()
    if now - _last_write_ts < RATE_LIMIT_INTERVAL:
        _queue_pending(text)
        return []
    _last_write_ts = now

    m = _get_mem0()
    if not m:
        _queue_pending(text)
        return []
    try:
        result = m.add(text, user_id=user_id, metadata=metadata or {})
        entries = result.get("results", [])
        if entries:
            log.info("Learned %d facts: %s", len(entries), text[:50])
            _prune_if_needed(user_id)
        return entries
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "429" in err:
            log.warning("Rate limited — queuing memory")
            _queue_pending(text)
        elif "connection" in err or "timeout" in err:
            log.warning("Offline — queuing memory")
            _queue_pending(text)
        else:
            log.error("Memory add failed: %s", e)
        return []


def recall(query: str, user_id: str = "user", limit: int = 8) -> list[dict]:
    """Search memories semantically."""
    m = _get_mem0()
    if not m:
        return []
    try:
        results = m.search(query, user_id=user_id, limit=limit)
        return results.get("results", [])
    except Exception as e:
        log.error("Recall failed: %s", e)
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
        log.error("Get all failed: %s", e)
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
        log.error("Delete failed: %s", e)


# ── Context for LLM ─────────────────────────────────────────────────────────

def get_context_for_llm(query: str, user_id: str = "user") -> str:
    """Get relevant memories for the assistant system prompt."""
    memories = recall(query, user_id=user_id, limit=8)
    if not memories:
        return ""
    lines = [f"- {m['memory']}" for m in memories if m.get("memory")]
    return "\n\nWhat I know about you:\n" + "\n".join(lines) if lines else ""


_names_cache: list[str] = []
_names_cache_ts: float = 0

def get_names_for_dictation(user_id: str = "user") -> list[str]:
    """Get known names/terms from memory for Whisper prompt hints. Cached 2 min."""
    global _names_cache, _names_cache_ts
    if time.time() - _names_cache_ts < 120 and _names_cache:
        return _names_cache
    memories = get_all(user_id)
    names = set()
    import re
    for m in memories:
        text = m.get("memory", "")
        # Extract capitalized words (likely names)
        for word in re.findall(r'\b[A-Z][a-z]+\b', text):
            if len(word) > 2:
                names.add(word)
    _names_cache = list(names)[:30]
    _names_cache_ts = time.time()
    return _names_cache


# ── Rate limiting + offline queue ────────────────────────────────────────────

def _queue_pending(text: str):
    """Queue text for later processing (rate limit / offline)."""
    with _queue_lock:
        if len(_pending_queue) < 50:  # cap queue
            _pending_queue.append(text)


def flush_pending(user_id: str = "user"):
    """Process queued memories (call periodically)."""
    global _last_write_ts
    with _queue_lock:
        items = _pending_queue.copy()
        _pending_queue.clear()
    for text in items:
        now = time.time()
        if now - _last_write_ts < RATE_LIMIT_INTERVAL:
            time.sleep(RATE_LIMIT_INTERVAL)
        remember(text, user_id=user_id)


# ── Growth management ────────────────────────────────────────────────────────

def _prune_if_needed(user_id: str = "user"):
    """Remove oldest memories if over the cap."""
    try:
        all_mems = get_all(user_id)
        if len(all_mems) <= MAX_MEMORIES:
            return
        # Sort by created_at, delete oldest
        sorted_mems = sorted(all_mems, key=lambda m: m.get("created_at", ""))
        to_delete = sorted_mems[:len(sorted_mems) - MAX_MEMORIES]
        for m in to_delete:
            delete(m["id"])
        log.info("Pruned %d old memories (total was %d)", len(to_delete), len(all_mems))
    except Exception as e:
        log.error("Prune failed: %s", e)


# ── Export / Import ──────────────────────────────────────────────────────────

def export_memories(user_id: str = "user") -> str:
    """Export all memories to JSON file. Returns path."""
    memories = get_all(user_id)
    data = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(memories),
        "memories": memories,
    }
    safe_json.save(EXPORT_PATH, data)
    log.info("Exported %d memories to %s", len(memories), EXPORT_PATH)
    return EXPORT_PATH


def import_memories(path: str, user_id: str = "user") -> int:
    """Import memories from a JSON export file. Returns count imported."""
    data = safe_json.load(path, {})
    memories = data.get("memories", [])
    count = 0
    for m in memories:
        text = m.get("memory", "")
        if text:
            remember(text, user_id=user_id)
            count += 1
    log.info("Imported %d memories from %s", count, path)
    return count


# ── Async helpers ────────────────────────────────────────────────────────────

def remember_async(text: str, user_id: str = "user", metadata: dict = None):
    """Store memory in background (non-blocking)."""
    if not text or len(text.strip()) < 10:
        return
    threading.Thread(target=remember, args=(text, user_id, metadata), daemon=True).start()
