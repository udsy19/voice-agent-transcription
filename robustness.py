"""Robustness infrastructure: persistent history, error tracking, health checks,
disk space, backups, rate limits, telemetry, crash dumps.

All in one module to keep the surface area small.
"""

import os
import json
import time
import shutil
import threading
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from config import DATA_DIR
from logger import get_logger
import safe_json

log = get_logger("robust")

# ── Paths ─────────────────────────────────────────────────────────────────
HISTORY_DIR = DATA_DIR / "history"
BACKUPS_DIR = DATA_DIR / "backups"
CRASHES_DIR = DATA_DIR / "crashes"
TELEMETRY_FILE = DATA_DIR / "telemetry.jsonl"
QUOTA_FILE = DATA_DIR / "quota.json"

for d in (HISTORY_DIR, BACKUPS_DIR, CRASHES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Persistent History (JSONL append-only) ────────────────────────────────

_history_lock = threading.Lock()


def _history_path() -> Path:
    """One file per day for easy rotation."""
    return HISTORY_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def append_history(entry: dict) -> bool:
    """Append a history entry to today's JSONL file. Survives crashes."""
    try:
        with _history_lock:
            with open(_history_path(), "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        return True
    except Exception as e:
        log.error("History append failed: %s", e)
        record_error("history_append", str(e))
        return False


def load_recent_history(limit: int = 100) -> list:
    """Load most recent N history entries across all daily files."""
    entries = []
    try:
        files = sorted(HISTORY_DIR.glob("*.jsonl"), reverse=True)
        for f in files:
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except IOError:
                continue
            if len(entries) >= limit:
                break
        # Sort newest-first by ts (already roughly ordered, but be safe)
        entries.sort(key=lambda e: e.get("ts", ""), reverse=True)
        return entries[:limit]
    except Exception as e:
        log.error("History load failed: %s", e)
        return []


def clear_history(keep_backup: bool = True) -> bool:
    """Delete all history files, optionally backing them up first."""
    try:
        if keep_backup:
            backup_dir = BACKUPS_DIR / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for f in HISTORY_DIR.glob("*.jsonl"):
                shutil.copy2(f, backup_dir / f.name)
        for f in HISTORY_DIR.glob("*.jsonl"):
            f.unlink()
        log.info("History cleared%s", " (backup saved)" if keep_backup else "")
        return True
    except Exception as e:
        log.error("Clear history failed: %s", e)
        return False


# ── Error Tracking (in-memory ring buffer) ────────────────────────────────

_errors = deque(maxlen=50)
_errors_lock = threading.Lock()


def record_error(category: str, message: str, details: str = ""):
    """Record an error to the in-memory ring buffer for /api/errors."""
    with _errors_lock:
        _errors.appendleft({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "category": category,
            "message": str(message)[:500],
            "details": str(details)[:1000],
        })


def get_errors() -> list:
    """Return all tracked errors, newest first."""
    with _errors_lock:
        return list(_errors)


def clear_errors():
    with _errors_lock:
        _errors.clear()


# ── Crash Reports ─────────────────────────────────────────────────────────

def write_crash_report(exc: Exception, context: dict = None) -> str:
    """Write a crash report with traceback + context. Returns path."""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CRASHES_DIR / f"crash_{ts}.txt"
        with open(path, "w") as f:
            f.write(f"Muse Crash Report\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Type: {type(exc).__name__}\n")
            f.write(f"Message: {exc}\n\n")
            if context:
                f.write("Context:\n")
                f.write(json.dumps(context, default=str, indent=2)[:4000])
                f.write("\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n\nRecent errors:\n")
            for err in list(_errors)[:10]:
                f.write(f"  {err['ts']} [{err['category']}] {err['message']}\n")
        log.error("Crash report written to %s", path)
        return str(path)
    except Exception as e:
        log.error("Could not write crash report: %s", e)
        return ""


def list_crash_reports(limit: int = 10) -> list:
    """List recent crash reports."""
    try:
        files = sorted(CRASHES_DIR.glob("crash_*.txt"), reverse=True)[:limit]
        return [{"path": str(f), "name": f.name,
                 "size": f.stat().st_size,
                 "ts": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
                for f in files]
    except Exception:
        return []


# ── Disk Space Checks ────────────────────────────────────────────────────

def check_disk_space(required_mb: int = 100) -> tuple[bool, int]:
    """Check if at least `required_mb` is available. Returns (ok, free_mb)."""
    try:
        stat = shutil.disk_usage(str(DATA_DIR))
        free_mb = stat.free // (1024 * 1024)
        return free_mb >= required_mb, free_mb
    except Exception:
        return True, -1  # assume ok if we can't check


# ── Backups Before Destructive Operations ────────────────────────────────

def backup_file(path: str | Path, keep_days: int = 7) -> str:
    """Backup a file before deleting/modifying it. Returns backup path."""
    src = Path(path)
    if not src.exists():
        return ""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = BACKUPS_DIR / f"{src.stem}_{ts}{src.suffix}"
        shutil.copy2(src, dst)
        # Cleanup old backups
        cutoff = time.time() - (keep_days * 86400)
        for old in BACKUPS_DIR.glob(f"{src.stem}_*"):
            try:
                if old.stat().st_mtime < cutoff:
                    old.unlink()
            except OSError:
                pass
        return str(dst)
    except Exception as e:
        log.error("Backup failed for %s: %s", path, e)
        return ""


# ── Health Checks ────────────────────────────────────────────────────────

_health_state = {
    "groq": {"ok": None, "msg": "Not checked", "ts": 0},
    "google": {"ok": None, "msg": "Not checked", "ts": 0},
    "local_llm": {"ok": None, "msg": "Not checked", "ts": 0},
    "kokoro": {"ok": None, "msg": "Not checked", "ts": 0},
    "disk": {"ok": None, "msg": "Not checked", "ts": 0},
}
_health_lock = threading.Lock()


def update_health(service: str, ok: bool, msg: str = ""):
    with _health_lock:
        _health_state[service] = {"ok": ok, "msg": msg, "ts": time.time()}


def get_health() -> dict:
    with _health_lock:
        return {k: dict(v) for k, v in _health_state.items()}


def run_health_checks(oauth_mgr=None):
    """Run all health checks. Call periodically (e.g. every 5min)."""
    # Disk
    ok, free_mb = check_disk_space(500)
    update_health("disk", ok, f"{free_mb} MB free" if free_mb >= 0 else "unknown")

    # Local LLM model files
    try:
        from model_manager import is_model_downloaded
        if is_model_downloaded("llm"):
            update_health("local_llm", True, "Mistral 7B installed")
        else:
            update_health("local_llm", False, "Not downloaded")
    except Exception as e:
        update_health("local_llm", False, f"Check failed: {e}"[:100])

    # Kokoro TTS files
    try:
        kokoro_path = DATA_DIR / "models" / "kokoro-v1.0.onnx"
        voices_path = DATA_DIR / "models" / "voices-v1.0.bin"
        if kokoro_path.exists() and voices_path.exists():
            update_health("kokoro", True, "TTS files OK")
        else:
            update_health("kokoro", False, "Will download on first use")
    except Exception:
        update_health("kokoro", False, "Check failed")

    # Groq reachability — only check if we have a key
    try:
        from config import GROQ_API_KEY
        if not GROQ_API_KEY:
            update_health("groq", None, "No API key")
        else:
            # Use httpx with certifi (already a dependency) — handles SSL properly
            import httpx, certifi
            try:
                with httpx.Client(verify=certifi.where(), timeout=5) as client:
                    r = client.get("https://api.groq.com/openai/v1/models",
                                   headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
                    if r.status_code == 200:
                        update_health("groq", True, "Connected")
                    elif r.status_code == 401:
                        update_health("groq", False, "Invalid API key")
                    else:
                        update_health("groq", False, f"HTTP {r.status_code}")
            except Exception as e:
                update_health("groq", False, f"Unreachable: {str(e)[:60]}")
    except Exception as e:
        update_health("groq", False, str(e)[:80])

    # Google OAuth
    if oauth_mgr:
        try:
            accts = oauth_mgr.list_accounts("google")
            if not accts:
                update_health("google", None, "Not connected")
            else:
                token = oauth_mgr.get_token("google", accts[0]["email"])
                if token:
                    update_health("google", True, accts[0]["email"])
                else:
                    update_health("google", False, "Token expired/revoked")
        except Exception as e:
            update_health("google", False, str(e)[:80])


# ── Rate Limiting / Quota Tracking ───────────────────────────────────────

_quota_lock = threading.Lock()


def _load_quota() -> dict:
    return safe_json.load(str(QUOTA_FILE), {"groq": {"day": "", "count": 0}})


def _save_quota(q: dict):
    safe_json.save(str(QUOTA_FILE), q)


def record_groq_call():
    """Track Groq API call count for the day."""
    with _quota_lock:
        q = _load_quota()
        today = datetime.now().strftime("%Y-%m-%d")
        if q.get("groq", {}).get("day") != today:
            q["groq"] = {"day": today, "count": 0}
        q["groq"]["count"] = q["groq"].get("count", 0) + 1
        _save_quota(q)


def get_groq_quota() -> dict:
    q = _load_quota()
    today = datetime.now().strftime("%Y-%m-%d")
    if q.get("groq", {}).get("day") != today:
        return {"day": today, "count": 0}
    return q.get("groq", {"day": today, "count": 0})


# ── Telemetry (opt-in, local-only by default) ────────────────────────────

_telemetry_enabled = False


def set_telemetry(enabled: bool):
    global _telemetry_enabled
    _telemetry_enabled = enabled


def telemetry(event: str, props: dict = None):
    """Record a telemetry event. Local-only — never sent anywhere."""
    if not _telemetry_enabled:
        return
    try:
        with open(TELEMETRY_FILE, "a") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "event": event,
                "props": props or {},
            }) + "\n")
    except Exception:
        pass


def get_telemetry_summary(limit: int = 1000) -> dict:
    """Aggregate telemetry events for display."""
    if not TELEMETRY_FILE.exists():
        return {"total": 0, "events": {}}
    counts = {}
    total = 0
    try:
        with open(TELEMETRY_FILE) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                try:
                    e = json.loads(line)
                    counts[e["event"]] = counts.get(e["event"], 0) + 1
                    total += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return {"total": total, "events": counts}


# ── LLM Output Schemas (Pydantic) ────────────────────────────────────────

try:
    from pydantic import BaseModel, ValidationError, Field
    from typing import List, Optional

    class MeetingSummarySchema(BaseModel):
        key_points: List[str] = Field(default_factory=list, max_length=20)
        action_items: List[str] = Field(default_factory=list, max_length=20)
        decisions: List[str] = Field(default_factory=list, max_length=20)

    class StandupSchema(BaseModel):
        yesterday: List[str] = Field(default_factory=list, max_length=10)
        today: List[str] = Field(default_factory=list, max_length=10)
        blockers: List[str] = Field(default_factory=list, max_length=10)

    def validate_meeting_summary(data: dict) -> dict:
        """Validate + sanitize meeting summary. Returns safe dict or empty fallback."""
        try:
            m = MeetingSummarySchema(**data)
            return {
                "key_points": [str(x)[:500] for x in m.key_points],
                "action_items": [str(x)[:500] for x in m.action_items],
                "decisions": [str(x)[:500] for x in m.decisions],
            }
        except (ValidationError, TypeError, ValueError) as e:
            log.warning("MeetingSummary validation failed: %s", e)
            return {"key_points": [], "action_items": [], "decisions": [], "_error": "invalid"}

    _has_pydantic = True
except ImportError:
    log.warning("Pydantic not available — skipping schema validation")
    _has_pydantic = False

    def validate_meeting_summary(data: dict) -> dict:
        # Fallback validation without pydantic
        out = {"key_points": [], "action_items": [], "decisions": []}
        if not isinstance(data, dict):
            return {**out, "_error": "not_dict"}
        for k in out:
            v = data.get(k, [])
            if isinstance(v, list):
                out[k] = [str(x)[:500] for x in v[:20] if x]
        return out


# ── Schema Migration ─────────────────────────────────────────────────────

SCHEMA_VERSION = 1
SCHEMA_FILE = DATA_DIR / "schema_version.json"


def get_schema_version() -> int:
    return safe_json.load(str(SCHEMA_FILE), {"version": 0}).get("version", 0)


def set_schema_version(v: int):
    safe_json.save(str(SCHEMA_FILE), {"version": v})


def run_migrations():
    """Run any pending data migrations."""
    current = get_schema_version()
    if current >= SCHEMA_VERSION:
        return
    log.info("Running migrations from v%d to v%d", current, SCHEMA_VERSION)
    # Migration v0 → v1: import any legacy in-memory history if present
    if current < 1:
        # Nothing to migrate — clean install or already on v1
        pass
    set_schema_version(SCHEMA_VERSION)
    log.info("Migrations complete (now v%d)", SCHEMA_VERSION)
