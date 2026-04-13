#!/usr/bin/env python3
"""Muse — web dashboard + voice backend.

Single command: python3 app.py
Opens http://localhost:8528 in your browser.
Global hotkeys work in the background.

Hold Right Option → record → release → transcribe → paste
"""

import sys
import os
import json
import threading
import time
import asyncio
import webbrowser
import subprocess

os.chdir("/tmp")
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
import safe_json


# ── Request Models ──────────────────────────────────────────────────────────

class DomainRequest(BaseModel):
    domain: str = ""

class MacroAddRequest(BaseModel):
    trigger: str
    description: str = ""
    actions: list[dict] = Field(default_factory=list)

class TriggerRequest(BaseModel):
    trigger: str

class TermRequest(BaseModel):
    term: str

class CorrectionRequest(BaseModel):
    wrong: str
    correct: str

class SnippetAddRequest(BaseModel):
    trigger: str
    text: str

class GroqKeyRequest(BaseModel):
    key: str

class BackendRequest(BaseModel):
    backend: str

class RoleRequest(BaseModel):
    role: str

from logger import get_logger
log = get_logger("app")

from pynput import keyboard
from config import SILENCE_THRESHOLD, DATA_DIR
from recorder import Recorder
from transcriber import Transcriber
from cleaner import Cleaner, _get_active_app
from injector import inject_text, undo_last_paste, get_selected_text
from dictionary import PersonalDictionary
from snippets import SnippetStore
from styles import StyleManager
from domains import DomainManager
from macros import MacroEngine
from conversation import ConversationTracker
from exporter import export_json, export_txt, export_word, export_pdf, export_logs, export_meeting_txt, export_meeting_pdf
from assistant import Assistant
from integrations.oauth_manager import OAuthManager
from todos import TodoList
from brain import Brain
import quick_capture
import briefing
import memory as mem
import tts
PORT = 8528
HANDS_FREE_SILENCE_SEC = 2.0

# ── Hotkey config ───────────────────────────────────────────────────────────
_HOTKEY_MAP = {
    "alt_r": keyboard.Key.alt_r,
    "alt_l": keyboard.Key.alt_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl_l": keyboard.Key.ctrl_l,
    "cmd_r": keyboard.Key.cmd_r,
    "fn": keyboard.Key.f20,  # Fn key sends F20 on some Macs
}

def _load_hotkey():
    """Load saved hotkey from config, default to Right Option."""
    try:
        import json
        hk_path = os.path.join(str(__import__('config').DATA_DIR), "hotkey.json")
        if os.path.exists(hk_path):
            with open(hk_path) as f:
                data = json.load(f)
            key_name = data.get("key", "alt_r")
            return _HOTKEY_MAP.get(key_name, keyboard.Key.alt_r), key_name
    except Exception:
        pass
    return keyboard.Key.alt_r, "alt_r"

TRIGGER_KEY, TRIGGER_KEY_NAME = _load_hotkey()

# Left Option = dictation, Right Option = assistant
DICTATION_KEY = keyboard.Key.alt_l   # ⌥L = dictate
ASSISTANT_KEY = keyboard.Key.alt_r   # ⌥R = assistant

# Undo keywords
UNDO_PHRASES = {"undo that", "go back", "undo", "scratch that", "never mind", "cancel that"}

# Apps where we should NOT inject text (system UIs, voice assistants)
BLOCKED_APPS = {"Siri", "Spotlight", "Alfred", "Raycast", "Launcher"}

# ── Sound feedback (barely noticeable system sounds) ────────────────────────
_SOUNDS = {
    "start": "/System/Library/Sounds/Tink.aiff",
    "done":  "/System/Library/Sounds/Pop.aiff",
    "error": "/System/Library/Sounds/Funk.aiff",
}
def _load_preferences() -> dict:
    """Load persistent user preferences."""
    import safe_json
    from config import DATA_DIR
    return safe_json.load(str(DATA_DIR / "preferences.json"), {
        "sounds": True,
        "clipboard_context": True,
    })

def _save_preferences():
    """Save current preferences to disk."""
    import safe_json
    from config import DATA_DIR
    safe_json.save(str(DATA_DIR / "preferences.json"), {
        "sounds": _sound_enabled,
        "clipboard_context": _clipboard_context_enabled,
    })

_prefs = _load_preferences()
_sound_enabled = _prefs.get("sounds", True)
_clipboard_context_enabled = _prefs.get("clipboard_context", True)

def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucinations — gibberish in random scripts when audio is unclear."""
    import unicodedata
    if not text or len(text.strip()) < 2:
        return True
    # Count characters by script category
    latin = 0
    non_latin = 0
    for ch in text:
        if ch.isalpha():
            cat = unicodedata.category(ch)
            # Check if character is Latin (basic Latin, Latin Extended, etc.)
            try:
                name = unicodedata.name(ch, '')
                if 'LATIN' in name or ch.isascii():
                    latin += 1
                else:
                    non_latin += 1
            except ValueError:
                non_latin += 1
    total = latin + non_latin
    if total == 0:
        return True
    # If more than 15% of alphabetic chars are non-Latin, it's likely hallucinated
    if total > 3 and non_latin / total > 0.15:
        return True
    # Multiple distinct non-Latin scripts = definite hallucination
    scripts = set()
    for ch in text:
        if ch.isalpha():
            try:
                n = unicodedata.name(ch, '')
                script = n.split()[0] if n else ''
                if script and script != 'LATIN':
                    scripts.add(script)
            except ValueError:
                pass
    if len(scripts) >= 2:
        return True
    # Check for known hallucination patterns
    hallucination_markers = [
        "thank you for watching", "thanks for watching", "subscribe",
        "please like and subscribe", "see you in the next",
        "subtitles by", "translated by", "copyright",
    ]
    text_lower = text.lower().strip()
    for marker in hallucination_markers:
        if marker in text_lower:
            return True
    return False


def _play_sound(name: str):
    """Play a subtle system sound in background. Non-blocking."""
    if not _sound_enabled:
        return
    path = _SOUNDS.get(name)
    if path:
        try:
            subprocess.Popen(
                ["afplay", "-v", "0.3", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


# ── Global State ─────────────────────────────────────────────────────────────

class State:
    """Centralized application state with proper initialization."""

    def __init__(self):
        # Engine components (initialized in load_engine)
        self.recorder: Recorder | None = None
        self.transcriber: Transcriber | None = None
        self.cleaner: Cleaner | None = None
        self.dictionary: PersonalDictionary | None = None
        self.snippets: SnippetStore | None = None
        self.styles: StyleManager | None = None
        self.domains: DomainManager | None = None
        self.macros: MacroEngine | None = None
        self.conversation: ConversationTracker | None = None
        self.assistant: Assistant | None = None
        self.oauth: OAuthManager | None = None
        self.todos: TodoList | None = None
        self.brain: Brain | None = None
        self.meeting_recorder = None
        self.meeting_detector = None

        # Runtime state
        self.status: str = "loading"
        self.detail: str = "Loading model..."
        self.processing: bool = False
        self.hands_free: bool = False
        self.whisper_mode: bool = False
        self.key_held: bool = False

        # Text state
        self.last_paste_text: str = ""
        self.last_paste_time: float = 0.0
        self.tone_override: str | None = None  # set by macros
        self.source_app: str = ""  # app that had focus when recording started
        self.recording_mode: str = "dictation"  # "dictation" or "assistant"

        # Collections
        self.history: list[dict] = []  # capped at 100 entries
        self.ws_clients: list[WebSocket] = []

    @property
    def is_ready(self) -> bool:
        return self.recorder is not None and self.transcriber is not None

    def reset_after_paste(self, text: str):
        """Update state after a successful paste."""
        self.last_paste_text = text
        self.last_paste_time = time.time()
        self.tone_override = None

S = State()


async def broadcast(msg: dict):
    dead = []
    for ws in list(S.ws_clients):  # iterate copy to avoid mutation during iteration
        try:
            await ws.send_json(msg)
        except Exception as e:
            log.debug("WebSocket send failed: %s", e)
            dead.append(ws)
    for ws in dead:
        try:
            S.ws_clients.remove(ws)
        except ValueError:
            pass  # already removed


def pill_notify(text: str, icon: str = "info", duration_ms: int = 3500):
    """Send a brief notification to the pill UI."""
    emit({"type": "pill_notify", "text": text, "icon": icon, "duration_ms": duration_ms})


def emit(msg: dict):
    loop = getattr(emit, '_loop', None)
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(msg), loop)


_history_lock = threading.Lock()

def add_history(entry):
    with _history_lock:
        S.history.append(entry)
        if len(S.history) > 100:
            S.history = S.history[-100:]


def set_status(status, detail=""):
    S.status = status
    S.detail = detail
    emit({"type": "status", "status": status, "detail": detail})


# ── FastAPI ──────────────────────────────────────────────────────────────────

api = FastAPI()

# Only allow requests from localhost (Electron and local browser)
from fastapi.middleware.cors import CORSMiddleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8528", "http://127.0.0.1:8528", "file://"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):852[89]",
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
)


@api.get("/")
async def index():
    return FileResponse(os.path.join(project_dir, "ui", "index.html"))


@api.get("/api/health")
async def health():
    return {"ok": True, "status": S.status}


@api.post("/api/start-recording")
async def api_start_recording():
    """Start recording — for pill click."""
    if S.recorder and not S.recorder.is_recording and not S.processing:
        S.source_app = _get_active_app()  # capture BEFORE recording starts
        S.recorder.start()
        set_status("recording", "Listening — click stop or release ⌥R")
        return {"ok": True}
    return {"ok": False}


@api.post("/api/stop-recording")
async def api_stop_recording():
    """Stop recording and process — for pill stop button."""
    if S.recorder and S.recorder.is_recording:
        stop_and_process()
        return {"ok": True}
    return {"ok": False, "reason": "not recording"}


@api.post("/api/cancel-recording")
async def api_cancel_recording():
    """Cancel recording, discard audio — for pill X button."""
    if S.recorder and S.recorder.is_recording:
        S.recorder.stop()  # discard audio
        S.key_held = False
        set_status("idle", "Recording cancelled")
        return {"ok": True}
    return {"ok": False, "reason": "not recording"}


@api.get("/api/state")
async def get_state():
    return {
        "status": S.status, "detail": S.detail,
        "hands_free": S.hands_free, "whisper_mode": S.whisper_mode,
        "history": S.history[-20:],
        "terms": S.dictionary.terms if S.dictionary else [],
        "corrections": S.dictionary.corrections if S.dictionary else {},
        "snippets": {k: v.get("text", "") for k, v in S.snippets._snippets.items()} if S.snippets else {},
        "role": S.styles._user_role if S.styles else "",
        "default_style": S.styles._default_style if S.styles else "",
        "app_overrides": S.styles._app_overrides if S.styles else {},
        "domains": S.domains.list_domains() if S.domains else {},
        "active_domain": S.domains.get_active() if S.domains else "",
        "transcription_backend": S.transcriber.backend if S.transcriber else "loading",
        "macros": S.macros.list_all() if S.macros else {},
    }


@api.get("/api/domains")
async def api_get_domains():
    if not S.domains:
        return {"domains": {}, "active": ""}
    return {
        "domains": {k: v.get("description", k) for k, v in S.domains._domains.items()},
        "active": S.domains.get_active(),
    }


@api.post("/api/domains/set")
async def api_set_domain(body: DomainRequest):
    d = body.domain.strip()
    if S.domains:
        S.domains.set_active(d)
    return {"ok": True, "active": d}


@api.get("/api/macros")
async def api_get_macros():
    return {"macros": S.macros.list_all() if S.macros else {}}


@api.post("/api/macros/add")
async def api_add_macro(body: MacroAddRequest):
    trigger = body.trigger.strip()
    if trigger and body.actions and S.macros:
        S.macros.add(trigger, body.description.strip(), body.actions)
    return {"ok": True}


@api.post("/api/macros/remove")
async def api_remove_macro(body: TriggerRequest):
    trigger = body.trigger.strip()
    if trigger and S.macros:
        S.macros.remove(trigger)
    return {"ok": True}


@api.post("/api/dictionary/add-term")
async def api_add_term(body: TermRequest):
    t = body.term.strip()
    if t and S.dictionary:
        S.dictionary.add_term(t)
    return {"ok": True}


@api.post("/api/dictionary/add-correction")
async def api_add_correction(body: CorrectionRequest):
    w, c = body.wrong.strip(), body.correct.strip()
    if w and c and S.dictionary:
        S.dictionary.add_correction(w, c)
    return {"ok": True}


@api.post("/api/dictionary/remove-term")
async def api_remove_term(body: TermRequest):
    t = body.term.strip()
    if t and S.dictionary:
        S.dictionary.remove_term(t)
    return {"ok": True}


@api.post("/api/snippets/add")
async def api_add_snippet(body: SnippetAddRequest):
    t, txt = body.trigger.strip(), body.text.strip()
    if t and txt and S.snippets:
        S.snippets.add(t, txt)
    return {"ok": True}


@api.post("/api/snippets/remove")
async def api_remove_snippet(body: TriggerRequest):
    t = body.trigger.strip()
    if t and S.snippets:
        S.snippets.remove(t)
    return {"ok": True}


@api.post("/api/set-groq-key")
async def api_set_groq_key(body: GroqKeyRequest):
    """Set Groq API key from the settings UI. Saves to .env and reinitializes cleaner."""
    import re as _re
    key = body.key.strip()
    if not key:
        return {"ok": False, "reason": "empty key"}
    # Validate key format (Groq keys are gsk_ followed by alphanumeric)
    if not _re.match(r'^gsk_[A-Za-z0-9_]{20,}$', key):
        return {"ok": False, "reason": "invalid key format — expected gsk_..."}
    import config
    # Store in Keychain (secure) + .env fallback
    config._keychain_set("Muse", "groq_api_key", key)
    # Also write .env for backward compat / dev mode
    from pathlib import Path
    from config import DATA_DIR
    from dotenv import set_key as _set_key
    env_path = DATA_DIR / ".env"
    env_path.touch(exist_ok=True)
    _set_key(str(env_path), "GROQ_API_KEY", key)
    try:
        dev_env = Path(__file__).parent / ".env"
        dev_env.touch(exist_ok=True)
        _set_key(str(dev_env), "GROQ_API_KEY", key)
    except Exception:
        pass
    config.GROQ_API_KEY = key
    from groq import Groq
    S.cleaner._client = Groq(api_key=key)
    log.info("Groq API key updated")
    return {"ok": True}


@api.get("/api/groq-status")
async def api_groq_status():
    from config import GROQ_API_KEY
    has_key = bool(GROQ_API_KEY)
    masked = GROQ_API_KEY[:4] + "***" + GROQ_API_KEY[-4:] if GROQ_API_KEY and len(GROQ_API_KEY) > 12 else "***" if GROQ_API_KEY else ""
    return {"configured": has_key, "masked_key": masked}


@api.post("/api/clear-groq-key")
async def api_clear_groq_key():
    """Remove the Groq API key from Keychain and .env files."""
    from utils import keychain_delete
    from pathlib import Path
    from config import DATA_DIR
    import config
    try:
        from dotenv import unset_key as _unset_key
    except Exception:
        _unset_key = None
    keychain_delete("Muse", "groq_api_key")
    for env_path in (DATA_DIR / ".env", Path(__file__).parent / ".env"):
        if env_path.exists() and _unset_key is not None:
            try:
                _unset_key(str(env_path), "GROQ_API_KEY")
            except Exception:
                pass
    config.GROQ_API_KEY = ""
    os.environ.pop("GROQ_API_KEY", None)
    # Reset cleaner's Groq client so it stops using the stale key
    try:
        S.cleaner._client = None
    except Exception:
        pass
    log.info("Groq API key cleared")
    return {"ok": True}


@api.get("/api/llm-mode")
async def api_llm_mode():
    """Get current LLM mode and local model status."""
    import llm as llm_mod
    from model_manager import is_model_downloaded
    provider = os.getenv("LLM_PROVIDER", "hybrid")
    local_ready = is_model_downloaded("llm")
    return {
        "mode": provider,
        "local_model_ready": local_ready,
        "local_model_name": llm_mod.LOCAL_MODEL.split("/")[-1] if local_ready else "",
    }


class LlmModeRequest(BaseModel):
    mode: str

_TRANSCRIPTION_FOR_MODE = {"local": "mlx", "groq": "groq", "hybrid": "groq"}

@api.post("/api/set-llm-mode")
async def api_set_llm_mode(body: LlmModeRequest):
    """Switch LLM provider mode + auto-switch transcription to match."""
    mode = body.mode.strip()
    if mode not in ("local", "groq", "hybrid"):
        return {"ok": False, "reason": "invalid mode"}
    os.environ["LLM_PROVIDER"] = mode
    import llm as llm_mod
    llm_mod._client = None
    # Auto-switch transcription backend to match
    suggested = _TRANSCRIPTION_FOR_MODE.get(mode, "groq")
    actual = suggested
    if S.transcriber:
        S.transcriber.set_backend(suggested)
        actual = S.transcriber.backend  # may differ if fallback occurred
    # Persist both (use actual, not suggested)
    prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
    prefs["llm_mode"] = mode
    prefs["transcription_backend"] = actual
    safe_json.save(str(DATA_DIR / "preferences.json"), prefs)
    log.info("LLM mode: %s, transcription: %s", mode, actual)
    return {"ok": True, "mode": mode, "transcription_backend": actual}


@api.get("/api/models/status")
async def api_models_status():
    from model_manager import get_models_status
    return {"models": get_models_status()}


class ModelDownloadRequest(BaseModel):
    model: str = "llm"

@api.post("/api/models/download")
async def api_models_download(body: ModelDownloadRequest):
    from model_manager import download_model_async
    download_model_async(body.model, emit_fn=emit)
    return {"ok": True, "model": body.model, "status": "downloading"}


@api.get("/api/export/{fmt}")
async def api_export(fmt: str):
    """Export history as JSON, TXT, Word (docx), or PDF."""
    import tempfile
    if not S.history:
        return {"ok": False, "reason": "no history to export"}
    ext_map = {"json": ".json", "txt": ".txt", "word": ".docx", "pdf": ".pdf"}
    if fmt not in ext_map:
        return {"ok": False, "reason": f"unknown format: {fmt}. Use: json, txt, word, pdf"}
    path = os.path.join(tempfile.gettempdir(), f"muse_export{ext_map[fmt]}")
    try:
        if fmt == "json":
            export_json(S.history, path)
        elif fmt == "txt":
            export_txt(S.history, path)
        elif fmt == "word":
            export_word(S.history, path)
        elif fmt == "pdf":
            export_pdf(S.history, path)
        return FileResponse(path, filename=f"muse-history{ext_map[fmt]}",
                           media_type="application/octet-stream")
    except ImportError as e:
        return {"ok": False, "reason": str(e)}
    except Exception as e:
        log.error("Export failed: %s", e)
        return {"ok": False, "reason": "export failed — check logs"}


@api.get("/api/search-history")
async def api_search_history(q: str = "", app: str = ""):
    """Search dictation history."""
    q_lower = q.lower()
    results = []
    for entry in reversed(S.history):
        text = (entry.get("cleaned", "") + " " + entry.get("raw", "")).lower()
        if q_lower and q_lower not in text:
            continue
        if app and entry.get("app", "").lower() != app.lower():
            continue
        results.append(entry)
    return {"results": results[:50], "total": len(results)}


@api.post("/api/clear-history")
async def api_clear_history():
    """Clear all dictation history."""
    with _history_lock:
        S.history.clear()
    if S.conversation:
        S.conversation.clear()
    return {"ok": True}


@api.post("/api/privacy/clipboard-context")
async def api_toggle_clipboard_context(body: dict):
    """Enable/disable clipboard context reading for Groq."""
    global _clipboard_context_enabled
    _clipboard_context_enabled = body.get("enabled", True)
    _save_preferences()
    return {"ok": True, "clipboard_context": _clipboard_context_enabled}


@api.get("/api/privacy")
async def api_privacy():
    """Get privacy settings."""
    return {
        "clipboard_context": _clipboard_context_enabled,
        "history_count": len(S.history),
        "sounds": _sound_enabled,
    }


@api.get("/api/export-logs")
async def api_export_logs():
    """Export log files as ZIP for bug reporting."""
    import tempfile
    log_dir = os.path.expanduser("~/Library/Logs/Muse")
    path = os.path.join(tempfile.gettempdir(), "muse_logs.zip")
    try:
        export_logs(log_dir, path)
        return FileResponse(path, filename="muse-logs.zip",
                           media_type="application/zip")
    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ── OAuth & Integrations ───────────────────────────────────────────────────

@api.post("/api/oauth/connect")
async def api_oauth_connect(body: dict = None):
    """Connect Google. If credentials not saved yet, pass client_id + client_secret."""
    if not S.oauth:
        return {"ok": False, "error": "Not initialized"}
    if body and body.get("client_id") and body.get("client_secret"):
        S.oauth.save_credentials(body["client_id"].strip(), body["client_secret"].strip())
    return S.oauth.connect(emit)


@api.get("/api/accounts")
async def api_list_accounts():
    """List all connected OAuth accounts."""
    accounts = S.oauth.list_accounts() if S.oauth else []
    return {"accounts": accounts}


@api.get("/api/oauth/status")
async def api_oauth_status():
    """Check OAuth connection status — used by frontend to detect expired tokens."""
    if not S.oauth:
        return {"connected": False, "email": "", "expired": False}
    accts = S.oauth.list_accounts("google")
    if not accts:
        return {"connected": False, "email": "", "expired": False}
    email = accts[0]["email"]
    # Try to get a token — if it returns None, token is expired/revoked
    token = S.oauth.get_token("google", email)
    return {"connected": token is not None, "email": email, "expired": token is None}


@api.delete("/api/accounts/{service}/{email}")
async def api_remove_account(service: str, email: str):
    """Disconnect an OAuth account."""
    if S.oauth:
        S.oauth.remove_account(service, email)
    return {"ok": True}


@api.get("/api/permissions")
async def api_permissions():
    """Check all macOS permissions with status and instructions."""
    import permissions
    from config import GROQ_API_KEY

    all_perms = permissions.check_all()
    ok, issues = permissions.check_required()

    return {
        "all_ok": ok,
        "permissions": all_perms,
        "issues": issues,
        "groq_key": bool(GROQ_API_KEY) or bool(S.cleaner and S.cleaner._client),
    }


@api.get("/api/memories")
async def api_memories():
    """Get all stored memories."""
    memories = mem.get_all()
    return {"memories": memories}


@api.post("/api/memories/search")
async def api_search_memories(body: dict):
    """Search memories."""
    query = body.get("query", "")
    if not query:
        return {"results": []}
    results = mem.recall(query, limit=10)
    return {"results": results}


@api.post("/api/memories/add")
async def api_add_memory(body: dict):
    """Manually add a memory."""
    text = body.get("text", "").strip()
    if not text:
        return {"ok": False}
    entries = mem.remember(text)
    return {"ok": True, "entries": entries}


@api.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: str):
    """Delete a memory."""
    mem.delete(memory_id)
    return {"ok": True}


@api.get("/api/memories/export")
async def api_export_memories():
    """Export all memories as JSON file."""
    path = mem.export_memories()
    return FileResponse(path, filename="muse-memories.json", media_type="application/json")


@api.post("/api/memories/import")
async def api_import_memories(body: dict):
    """Import memories from JSON data."""
    memories = body.get("memories", [])
    count = 0
    for m in memories:
        text = m.get("memory", "")
        if text:
            mem.remember(text)
            count += 1
    return {"ok": True, "imported": count}


@api.get("/api/todos")
async def api_todos():
    return {"todos": S.todos.list_all() if S.todos else []}

@api.post("/api/todos/add")
async def api_add_todo(body: dict):
    text = body.get("text", "").strip()
    if not text or not S.todos:
        return {"ok": False}
    item = S.todos.add(text)
    emit({"type": "todo_added", "item": item})
    return {"ok": True, "item": item}

@api.post("/api/todos/complete")
async def api_complete_todo(body: dict):
    tid = body.get("id", "")
    if S.todos and S.todos.complete(tid):
        emit({"type": "todo_completed", "id": tid})
        return {"ok": True}
    return {"ok": False}

@api.post("/api/todos/remove")
async def api_remove_todo(body: dict):
    tid = body.get("id", "")
    if S.todos and S.todos.remove(tid):
        return {"ok": True}
    return {"ok": False}

@api.get("/api/briefing")
async def api_briefing():
    """Get today's briefing."""
    events = []
    try:
        token = S.oauth.get_token("google") if S.oauth else None
        if token:
            from integrations.google_calendar import list_events
            r = list_events(token, days_ahead=1)
            events = r.get("events", []) if r.get("ok") else []
    except Exception:
        pass
    todos = S.todos.list_pending() if S.todos else []
    deadlines = S.brain.get_deadlines() if S.brain else []
    facts = S.brain.facts[-5:] if S.brain else []
    text = briefing.compose(events, todos, deadlines, facts)
    return {"text": text}

@api.post("/api/briefing/speak")
async def api_speak_briefing():
    """Speak the briefing aloud."""
    events = []
    try:
        token = S.oauth.get_token("google") if S.oauth else None
        if token:
            from integrations.google_calendar import list_events
            r = list_events(token, days_ahead=1)
            events = r.get("events", []) if r.get("ok") else []
    except Exception:
        pass
    todos = S.todos.list_pending() if S.todos else []
    deadlines = S.brain.get_deadlines() if S.brain else []
    text = briefing.compose(events, todos, deadlines, [])
    emit({"type": "assistant_stream", "text": text, "done": True})
    import safe_json
    prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
    voice = prefs.get("voice", "af_heart")
    threading.Thread(target=tts.speak_sync, args=(text, voice), daemon=True).start()
    return {"ok": True, "text": text}

@api.get("/api/follow-ups")
async def api_follow_ups():
    """Get pending follow-up reminders."""
    import follow_ups as fu
    return {"pending": fu.get_pending_reminders(), "all": fu.get_all()}


@api.get("/api/home/next-meeting")
async def api_next_meeting():
    """Get the next meeting starting within 30 minutes, with context."""
    if not S.oauth:
        return {"meeting": None}
    try:
        token = S.oauth.get_token("google")
        if not token:
            return {"meeting": None}
        from integrations.google_calendar import list_events
        from datetime import datetime, timezone
        r = list_events(token, days_ahead=1, max_results=10)
        if not r.get("ok"):
            return {"meeting": None}
        now = datetime.now(timezone.utc)
        for ev in r.get("events", []):
            try:
                start = datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))
                diff = (start - now).total_seconds()
                if 0 < diff <= 1800:  # within 30 min
                    context = ""
                    if S.brain:
                        context = S.brain.get_meeting_context(ev.get("summary", "")) or ""
                    return {"meeting": ev, "minutes_until": round(diff / 60), "past_context": context}
            except (ValueError, KeyError):
                continue
    except Exception:
        pass
    return {"meeting": None}


@api.get("/api/today")
async def api_today():
    """Get today's calendar events for the home card."""
    if not S.oauth:
        return {"events": []}
    try:
        token = S.oauth.get_token("google")
        if not token:
            return {"events": []}
        from integrations.google_calendar import list_events
        result = list_events(token, days_ahead=1, max_results=15)
        return {"events": result.get("events", [])} if result.get("ok") else {"events": []}
    except Exception:
        return {"events": []}


# ── Meetings ──────────────────────────────────────────────────────────────

@api.get("/api/meetings")
async def api_list_meetings():
    from meeting_recorder import list_meetings
    return {"meetings": list_meetings()}


@api.get("/api/meetings/status")
async def api_meeting_status():
    if S.meeting_recorder:
        return S.meeting_recorder.get_status()
    return {"recording": False}


@api.get("/api/meetings/upcoming")
async def api_meetings_upcoming():
    """Calendar events with meeting links in the next 7 days."""
    if not S.oauth:
        return {"events": []}
    try:
        token = S.oauth.get_token("google")
        if not token:
            return {"events": []}
        from integrations.google_calendar import list_events
        result = list_events(token, days_ahead=7, max_results=20)
        if not result.get("ok"):
            return {"events": []}
        events = [e for e in result.get("events", []) if e.get("meet_link") or "zoom" in (e.get("location", "") + e.get("description", "")).lower()]
        return {"events": events}
    except Exception:
        return {"events": []}


@api.post("/api/meetings/detect")
async def api_toggle_detection(body: dict):
    """Enable/disable meeting auto-detection."""
    enable = body.get("enabled", True)
    if not S.meeting_detector:
        return {"ok": False}
    if enable:
        S.meeting_detector.start_polling()
    else:
        S.meeting_detector.stop_polling()
    return {"ok": True, "detecting": S.meeting_detector._polling}


@api.get("/api/meetings/detect")
async def api_detection_status():
    return {"detecting": S.meeting_detector._polling if S.meeting_detector else False}


@api.get("/api/meetings/blackhole")
async def api_blackhole_status():
    from utils import detect_blackhole
    idx, name = detect_blackhole()
    return {"installed": idx is not None, "device_name": name}


@api.get("/api/meetings/{meeting_id}")
async def api_get_meeting(meeting_id: str):
    from meeting_recorder import get_meeting
    data = get_meeting(meeting_id)
    if not data:
        return {"ok": False, "reason": "not found"}
    return data


@api.post("/api/meetings/start")
async def api_start_meeting(body: dict):
    if not S.meeting_recorder:
        return {"ok": False, "reason": "not initialized"}
    if S.meeting_recorder.is_recording:
        return {"ok": False, "reason": "already recording", "meeting_id": S.meeting_recorder._meeting_id}
    title = body.get("title", "Meeting")
    cal_id = body.get("calendar_event_id")
    mid = S.meeting_recorder.start(title=title, calendar_event_id=cal_id)
    return {"ok": True, "meeting_id": mid}


@api.post("/api/meetings/stop")
async def api_stop_meeting():
    if not S.meeting_recorder or not S.meeting_recorder.is_recording:
        return {"ok": False, "reason": "not recording"}
    data = S.meeting_recorder.stop()
    # Check if meeting has calendar event with attendees → offer email draft
    cal_id = data.get("calendar_event_id") if data else None
    if cal_id and S.oauth:
        try:
            token = S.oauth.get_token("google")
            if token:
                from integrations.google_calendar import list_events
                r = list_events(token, days_ahead=1, max_results=20)
                for ev in r.get("events", []):
                    if ev.get("id") == cal_id and ev.get("attendees"):
                        emails = [a for a in ev["attendees"] if "@" in a and "self" not in a.lower()]
                        if emails:
                            emit({"type": "meeting_email_prompt",
                                  "meeting_id": data["id"], "meeting_title": data["title"],
                                  "attendees": emails})
                        break
        except Exception:
            pass
    return {"ok": True, "meeting": data}


@api.delete("/api/meetings/{meeting_id}")
async def api_delete_meeting(meeting_id: str):
    from meeting_recorder import delete_meeting
    return {"ok": delete_meeting(meeting_id)}


@api.post("/api/meetings/{meeting_id}/draft-email")
async def api_draft_meeting_email(meeting_id: str, body: dict):
    """Draft a follow-up email with the meeting summary to attendees."""
    to_list = body.get("to", [])
    if not to_list or not S.oauth:
        return {"ok": False, "reason": "no recipients or Google not connected"}
    token = S.oauth.get_token("google")
    if not token:
        return {"ok": False, "reason": "Google token expired"}
    from meeting_recorder import get_meeting
    m = get_meeting(meeting_id)
    if not m:
        return {"ok": False, "reason": "meeting not found"}
    s = m.get("summary", {})
    body_text = f"Hi,\n\nHere's a summary of our meeting: {m['title']}\n\n"
    if s.get("key_points"):
        body_text += "Key Points:\n" + "\n".join(f"  - {p}" for p in s["key_points"]) + "\n\n"
    if s.get("action_items"):
        body_text += "Action Items:\n" + "\n".join(f"  - {p}" for p in s["action_items"]) + "\n\n"
    if s.get("decisions"):
        body_text += "Decisions:\n" + "\n".join(f"  - {p}" for p in s["decisions"]) + "\n\n"
    body_text += "Best regards"
    from integrations.gmail import draft_email
    to = ", ".join(to_list)
    return draft_email(token, to, f"Meeting Notes: {m['title']}", body_text)


@api.get("/api/meetings/{meeting_id}/export/{fmt}")
async def api_export_meeting(meeting_id: str, fmt: str):
    """Export a single meeting as TXT or PDF."""
    from meeting_recorder import get_meeting
    import tempfile
    m = get_meeting(meeting_id)
    if not m:
        return {"ok": False, "reason": "not found"}
    if fmt not in ("txt", "pdf"):
        return {"ok": False, "reason": "use txt or pdf"}
    ext = ".txt" if fmt == "txt" else ".pdf"
    slug = (m.get("title", "meeting"))[:20].replace(" ", "_")
    path = os.path.join(tempfile.gettempdir(), f"muse_{slug}{ext}")
    try:
        if fmt == "txt":
            export_meeting_txt(m, path)
        else:
            export_meeting_pdf(m, path)
        return FileResponse(path, filename=f"muse-{slug}{ext}", media_type="application/octet-stream")
    except Exception as e:
        log.error("Meeting export failed: %s", e)
        return {"ok": False, "reason": "export failed"}


@api.post("/api/meetings/ask")
async def api_ask_meeting(body: dict):
    """Chat about a meeting — multi-turn conversation with transcript as context."""
    question = body.get("question", "").strip()
    meeting_id = body.get("meeting_id", "")
    history = body.get("history", [])  # prior Q&A turns
    if not question:
        return {"ok": False, "answer": ""}
    # Load meeting transcript from disk for full context
    transcript = ""
    if meeting_id:
        from meeting_recorder import get_meeting
        m = get_meeting(meeting_id)
        if m:
            transcript = "\n".join(f"[{c['timestamp']}] {c['speaker']}: {c['text']}" for c in m.get("chunks", []))
            summary = m.get("summary", {})
    try:
        from llm import get_client
        client = get_client()
        system_ctx = f"""You are a meeting assistant. You have full context of this meeting.
Answer questions conversationally. Be concise but thorough. Reference specific parts of the discussion when relevant.

Meeting transcript:
{transcript[:6000]}"""
        if summary:
            system_ctx += f"\n\nSummary: {json.dumps(summary)}"
        messages = [{"role": "system", "content": system_ctx}]
        # Add conversation history for multi-turn
        for h in history[-10:]:  # keep last 10 turns
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": question})
        resp = client.chat(messages=messages, model_tier="small", temperature=0.3, max_tokens=512)
        return {"ok": True, "answer": resp.text}
    except Exception as e:
        log.error("Meeting ask failed: %s", e)
        return {"ok": False, "answer": "Failed to process question"}


@api.get("/api/voices")
async def api_voices():
    """List all available TTS voices."""
    voices = tts.get_voices()
    import safe_json
    from config import DATA_DIR
    prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
    current = prefs.get("voice", "af_heart")
    return {"voices": voices, "current": current, "kokoro_available": tts.is_available()}


    # ElevenLabs removed — Kokoro only


@api.post("/api/voices/set")
async def api_set_voice(body: dict):
    """Set the TTS voice."""
    voice = body.get("voice", "").strip()
    if voice not in tts.get_voices():
        return {"ok": False, "error": "Unknown voice"}
    import safe_json
    from config import DATA_DIR
    prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
    prefs["voice"] = voice
    safe_json.save(str(DATA_DIR / "preferences.json"), prefs)
    return {"ok": True, "voice": voice}


@api.get("/api/tts/speed")
async def api_get_tts_speed():
    return {"speed": tts.tts_speed}


@api.post("/api/tts/speed")
async def api_set_tts_speed(body: dict):
    speed = float(body.get("speed", 1.0))
    speed = max(0.5, min(2.0, speed))
    tts.tts_speed = speed
    import safe_json
    from config import DATA_DIR
    prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
    prefs["tts_speed"] = speed
    safe_json.save(str(DATA_DIR / "preferences.json"), prefs)
    return {"ok": True, "speed": speed}


@api.post("/api/voices/preview")
async def api_preview_voice(body: dict):
    """Play a preview of a TTS voice."""
    voice = body.get("voice", "af_heart")
    text = body.get("text", tts.PREVIEW_TEXT)
    threading.Thread(target=tts.speak_sync, args=(text, voice), daemon=True).start()
    return {"ok": True}


@api.post("/api/voices/stop")
async def api_stop_voice():
    tts.stop()
    return {"ok": True}


@api.get("/api/debug")
async def api_debug():
    """Debug info for troubleshooting."""
    import sys
    return {
        "python": sys.executable,
        "recording_mode": S.recording_mode,
        "key_held": S.key_held,
        "processing": S.processing,
        "is_recording": S.recorder.is_recording if S.recorder else False,
        "dictation_key": str(DICTATION_KEY),
        "assistant_key": str(ASSISTANT_KEY),
        "source_app": S.source_app,
    }


@api.get("/api/transcription-backend")
async def api_get_backend():
    backend = S.transcriber.backend if S.transcriber else "loading"
    return {"backend": backend, "available": ["parakeet", "mlx", "groq", "faster-whisper"]}


@api.post("/api/transcription-backend")
async def api_set_backend(body: BackendRequest):
    backend = body.backend.strip()
    if backend and S.transcriber:
        S.transcriber.set_backend(backend)
        prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
        prefs["transcription_backend"] = backend
        safe_json.save(str(DATA_DIR / "preferences.json"), prefs)
        return {"ok": True, "backend": S.transcriber.backend}
    return {"ok": False}


@api.post("/api/styles/set-role")
async def api_set_role(body: RoleRequest):
    r = body.role.strip()
    if r and S.styles:
        S.styles.setup_role(r)
    return {"ok": True}


@api.post("/api/toggle-hands-free")
async def api_toggle_hf():
    toggle_hands_free()
    return {"ok": True}


@api.get("/api/hotkey")
async def api_get_hotkey():
    return {"key": TRIGGER_KEY_NAME, "available": list(_HOTKEY_MAP.keys())}


@api.post("/api/hotkey")
async def api_set_hotkey(body: dict):
    global TRIGGER_KEY, TRIGGER_KEY_NAME
    key_name = body.get("key", "").strip()
    if key_name not in _HOTKEY_MAP:
        return {"ok": False, "reason": f"unknown key: {key_name}"}
    TRIGGER_KEY = _HOTKEY_MAP[key_name]
    TRIGGER_KEY_NAME = key_name
    # Save to config
    from config import DATA_DIR
    import safe_json
    safe_json.save(os.path.join(str(DATA_DIR), "hotkey.json"), {"key": key_name})
    log.info("Hotkey changed to: %s", key_name)
    return {"ok": True, "key": key_name}


@api.post("/api/toggle-sounds")
async def api_toggle_sounds():
    global _sound_enabled
    _sound_enabled = not _sound_enabled
    _save_preferences()
    return {"ok": True, "sounds": _sound_enabled}


@api.post("/api/toggle-whisper")
async def api_toggle_w():
    if S.recorder:
        S.whisper_mode = S.recorder.toggle_whisper_mode()
        emit({"type": "whisper", "on": S.whisper_mode})
    return {"ok": True}


@api.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    S.ws_clients.append(ws)
    emit._loop = asyncio.get_event_loop()
    try:
        await ws.send_json({"type": "status", "status": S.status, "detail": S.detail})
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        if ws in S.ws_clients:
            S.ws_clients.remove(ws)


# ── Voice Engine ─────────────────────────────────────────────────────────────

_key_lock = threading.Lock()


_key_held_since = 0.0
_last_dictation_release = 0.0  # for double-tap detection
_dictation_toggle = False      # True = recording in toggle mode
DOUBLE_TAP_WINDOW = 0.35

def on_key_press(key):
    """⌥L = dictation (hold or double-tap toggle). ⌥R = assistant (hold only)."""
    global _key_held_since, _dictation_toggle

    if key == DICTATION_KEY:
        mode = "dictation"
    elif key == ASSISTANT_KEY:
        mode = "assistant"
    else:
        return

    with _key_lock:
        # Safety: reset stuck states after 30s
        if S.key_held and (time.time() - _key_held_since) > 30:
            log.warning("Resetting stuck key_held state")
            S.key_held = False
        if S.processing and (time.time() - _key_held_since) > 120:
            log.warning("Resetting stuck processing state")
            S.processing = False

        if S.processing:
            return

        # If in toggle mode and user presses ⌥L again → stop
        if mode == "dictation" and _dictation_toggle and S.recorder and S.recorder.is_recording:
            _dictation_toggle = False
            stop_and_process()
            return

        if S.key_held or S.hands_free:
            return
        if not S.is_ready:
            return

        S.key_held = True
        _key_held_since = time.time()
        S.recording_mode = mode
        if not S.recorder.is_recording:
            S.source_app = _get_active_app()
            S.recorder.start()
            _play_sound("start")
            emit({"type": "trigger_mode", "mode": "hold"})
            label = "assistant" if mode == "assistant" else "dictation"
            set_status("recording", f"Listening ({label}) — release to process")


def on_key_release(key):
    global _last_dictation_release, _dictation_toggle

    if key not in (DICTATION_KEY, ASSISTANT_KEY):
        return
    with _key_lock:
        if S.hands_free:
            toggle_hands_free()
            return

        now = time.time()

        if S.key_held:
            S.key_held = False
            held_duration = now - _key_held_since

            # ⌥L: if released quickly (< 300ms) and second tap within window → toggle mode
            if key == DICTATION_KEY and held_duration < 0.3:
                if (now - _last_dictation_release) < DOUBLE_TAP_WINDOW:
                    # Double tap! Switch to toggle mode — keep recording
                    _dictation_toggle = True
                    _last_dictation_release = 0
                    emit({"type": "trigger_mode", "mode": "toggle"})
                    set_status("recording", "Listening — press ⌥L to stop")
                    return
                else:
                    _last_dictation_release = now

            if S.recorder and S.recorder.is_recording and not _dictation_toggle:
                stop_and_process()


def toggle_hands_free():
    S.hands_free = not S.hands_free
    emit({"type": "hands_free", "on": S.hands_free})
    if S.hands_free:
        set_status("hands_free", "Speak freely — press ⌥R to stop")
        if S.recorder:
            S.recorder.start()
        threading.Thread(target=hands_free_loop, daemon=True).start()
    else:
        set_status("idle", "Ready")
        if S.recorder and S.recorder.is_recording:
            stop_and_process()


def hands_free_loop():
    import numpy as np
    silence_start = None
    while S.hands_free and S.recorder and S.recorder.is_recording:
        time.sleep(0.3)
        try:
            with S.recorder._lock:
                if not S.recorder._frames:
                    continue
                recent = S.recorder._frames[-1].copy().flatten()
                frame_count = len(S.recorder._frames)
        except Exception:
            continue
        rms = np.sqrt(np.mean(recent ** 2))
        threshold = SILENCE_THRESHOLD * 0.5 if S.whisper_mode else SILENCE_THRESHOLD
        if rms < threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= HANDS_FREE_SILENCE_SEC:
                if frame_count > 5:
                    audio = S.recorder.stop()
                    if audio is not None:
                        S.processing = True
                        process_audio(audio)
                    if S.hands_free and S.recorder:
                        S.recorder.start()
                    silence_start = None
        else:
            silence_start = None


def stop_and_process():
    try:
        audio = S.recorder.stop()
    except Exception as e:
        log.error("Recorder stop failed: %s", e)
        set_status("idle", "Recording error")
        S.key_held = False
        return
    if audio is None:
        set_status("idle", "Too short or silent")
        return
    S.processing = True

    def process_with_timeout():
        try:
            process_audio(audio)
        except Exception as e:
            log.error("process_audio crashed: %s", e, exc_info=True)
            set_status("idle", "Error processing audio")
        finally:
            S.processing = False
            S.recording_mode = "dictation"  # reset mode

    t = threading.Thread(target=process_with_timeout, daemon=True)
    t.start()

    # Safety watchdog: if processing takes >90s, force reset
    def watchdog():
        t.join(timeout=90)
        if t.is_alive():
            log.error("Processing stuck for >90s — force resetting state")
            S.processing = False
            S.recording_mode = "dictation"
            S.key_held = False
            set_status("idle", "Timed out — try again")
    threading.Thread(target=watchdog, daemon=True).start()



def _get_clipboard_context() -> str:
    """Read current clipboard for context injection into Groq."""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, timeout=1)
        try:
            text = result.stdout.decode('utf-8').strip()
        except UnicodeDecodeError:
            return ""
        if not text or len(text) > 2000:
            return ""
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
        return text[:500]
    except Exception:
        return ""


def _compute_diff(raw: str, cleaned: str) -> list[dict]:
    """Compute word-level diff between raw and cleaned text.

    Returns list of {type: "same"|"removed"|"added", text: str}
    """
    import difflib
    raw_words = raw.split()
    cleaned_words = cleaned.split()
    diff = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, raw_words, cleaned_words).get_opcodes():
        if tag == "equal":
            diff.append({"type": "same", "text": " ".join(raw_words[i1:i2])})
        elif tag == "delete":
            diff.append({"type": "removed", "text": " ".join(raw_words[i1:i2])})
        elif tag == "insert":
            diff.append({"type": "added", "text": " ".join(cleaned_words[j1:j2])})
        elif tag == "replace":
            diff.append({"type": "removed", "text": " ".join(raw_words[i1:i2])})
            diff.append({"type": "added", "text": " ".join(cleaned_words[j1:j2])})
    return diff


def process_audio(audio):
    try:
        t0 = time.time()
        # Use the app that was active when recording STARTED, not now
        app_name = S.source_app or _get_active_app()

        # Skip injection for blocked apps (Siri, Spotlight, etc.)
        if app_name in BLOCKED_APPS:
            log.info("Blocked app '%s' — skipping injection", app_name)

        set_status("processing", "Transcribing...")

        # Build whisper prompt from dictionary + domain
        prompt_parts = []
        if S.dictionary:
            dp = S.dictionary.get_whisper_prompt()
            if dp:
                prompt_parts.append(dp)
        if S.domains:
            ddp = S.domains.get_whisper_prompt()
            if ddp:
                prompt_parts.append(ddp)
        # Add names from memory for better Whisper spelling
        try:
            mem_names = mem.get_names_for_dictation()
            if mem_names:
                prompt_parts.append(", ".join(mem_names))
        except Exception:
            pass
        whisper_prompt = ", ".join(prompt_parts) if prompt_parts else None

        # Streaming transcription — emit partial results as segments complete
        def on_segment(partial_text):
            emit({"type": "partial", "text": partial_text})

        raw_text = S.transcriber.transcribe_streaming(
            audio, on_segment=on_segment, initial_prompt=whisper_prompt
        )
        if not raw_text:
            set_status("idle", "Nothing detected")
            return

        # Filter Whisper hallucinations (random scripts when audio is unclear)
        if _is_hallucination(raw_text):
            log.info("Filtered hallucination: %s", raw_text[:60])
            set_status("idle", "Unclear audio — try again")
            _play_sound("error")
            return

        # === Assistant mode (Right Option key) ===
        if S.recording_mode == "assistant" and S.assistant:
            set_status("processing", "Thinking...")
            response = S.assistant.handle(raw_text)
            dur = time.time() - t0
            entry = {"raw": raw_text, "cleaned": response or "", "app": app_name,
                     "duration": round(dur, 2), "ts": time.strftime("%H:%M:%S"),
                     "type": "assistant"}
            add_history(entry)
            emit({"type": "result", **entry})
            _play_sound("done")
            set_status("idle", f"Done ({dur:.1f}s)")
            return

        # === Quick Capture (dictation mode only) ===
        if S.recording_mode == "dictation":
            capture = quick_capture.detect(raw_text)
            if capture:
                intent, extracted = capture
                if intent == "todo" and S.todos:
                    item = S.todos.add(extracted)
                    emit({"type": "todo_added", "item": item})
                    _play_sound("done")
                    set_status("idle", f"Todo added: {extracted[:40]}")
                    entry = {"raw": raw_text, "cleaned": f"[todo] {extracted}", "app": app_name,
                             "duration": round(time.time() - t0, 2), "ts": time.strftime("%H:%M:%S")}
                    add_history(entry)
                    emit({"type": "result", **entry})
                    return
                elif intent == "memory":
                    mem.remember(extracted)
                    _play_sound("done")
                    set_status("idle", f"Noted: {extracted[:40]}")
                    entry = {"raw": raw_text, "cleaned": f"[note] {extracted}", "app": app_name,
                             "duration": round(time.time() - t0, 2), "ts": time.strftime("%H:%M:%S")}
                    add_history(entry)
                    emit({"type": "result", **entry})
                    return
                elif intent == "meeting_notes" and S.brain:
                    # Extract action items via simple split
                    S.brain.add_meeting_notes("Voice note", time.strftime("%Y-%m-%d"), extracted, [])
                    _play_sound("done")
                    set_status("idle", "Meeting notes saved")
                    entry = {"raw": raw_text, "cleaned": f"[meeting] {extracted}", "app": app_name,
                             "duration": round(time.time() - t0, 2), "ts": time.strftime("%H:%M:%S")}
                    add_history(entry)
                    emit({"type": "result", **entry})
                    return
                elif intent == "standup":
                    import standup
                    structured = standup.generate(extracted or raw_text)
                    inject_text(structured, target_app=app_name)
                    if S.brain:
                        S.brain.add_meeting_notes("Standup", time.strftime("%Y-%m-%d"), structured, [])
                    _play_sound("done")
                    set_status("idle", "Standup generated")
                    entry = {"raw": raw_text, "cleaned": structured, "app": app_name,
                             "duration": round(time.time() - t0, 2), "ts": time.strftime("%H:%M:%S")}
                    add_history(entry)
                    emit({"type": "result", **entry})
                    return
                elif intent == "calendar" and S.assistant:
                    # Route to assistant for calendar handling
                    pass  # fall through to assistant handling below

        # === Smart Undo ===
        raw_lower = raw_text.lower().strip()
        if raw_lower in UNDO_PHRASES and S.last_paste_text and (time.time() - S.last_paste_time) < 10:
            log.info("Undo triggered: '%s'", raw_lower)
            undo_last_paste()
            entry = {"raw": raw_text, "cleaned": "[undo]", "app": app_name,
                     "duration": round(time.time() - t0, 2), "ts": time.strftime("%H:%M:%S")}
            add_history(entry)
            emit({"type": "result", **entry})
            set_status("idle", "Undone")
            return

        # === Command mode ===
        is_command = raw_lower.startswith("hey flow")
        if is_command:
            raw_text = raw_text[8:].strip().lstrip(",").lstrip(".").strip()

        # === Macros ===
        if S.macros:
            macro = S.macros.match(raw_text)
            if macro:
                log.info("Macro triggered: %s", raw_text)
                context = {
                    "set_tone": lambda t: setattr(S, "tone_override", t),
                    "set_domain": lambda d: S.domains.set_active(d) if S.domains else None,
                    "inject_text": lambda t: inject_text(t),
                    "get_app": lambda: app_name,
                }
                results = S.macros.execute(macro, context)
                entry = {"raw": raw_text, "cleaned": f"[macro: {len(results)} actions]", "app": app_name,
                         "duration": round(time.time() - t0, 2), "ts": time.strftime("%H:%M:%S")}
                add_history(entry)
                emit({"type": "result", **entry})
                set_status("idle", f"Macro executed")
                return

        # === Snippets ===
        if S.snippets:
            snippet_text = S.snippets.match(raw_text)
            if snippet_text:
                inject_text(snippet_text, target_app=app_name)
                S.reset_after_paste(snippet_text)
                dur = time.time() - t0
                entry = {"raw": raw_text, "cleaned": "[snippet]", "app": app_name,
                         "duration": round(dur, 2), "ts": time.strftime("%H:%M:%S")}
                add_history(entry)
                emit({"type": "result", **entry})
                set_status("idle", f"Snippet inserted ({dur:.1f}s)")
                return

        # Apply dictionary corrections
        if S.dictionary:
            raw_text = S.dictionary.apply(raw_text)

        set_status("processing", "Cleaning...")

        # === Get clipboard context for better Groq accuracy ===
        clipboard_context = _get_clipboard_context() if _clipboard_context_enabled else ""

        # === Get domain hint ===
        domain_hint = S.domains.get_cleaner_hint() if S.domains else None

        # === Get conversation context ===
        conv_context = S.conversation.get_context(app_name) if S.conversation else None

        # === Build extra context for cleaner ===
        extra_context = ""
        if clipboard_context:
            extra_context += clipboard_context
        if domain_hint:
            extra_context += f"\n{domain_hint}" if extra_context else domain_hint
        if conv_context:
            extra_context += f"\n{conv_context}" if extra_context else conv_context

        # === Get style prompt from StyleManager ===
        style_prompt = S.styles.get_style_prompt(app_name) if S.styles else None

        if is_command:
            # Try assistant first (calendar, email, etc.)
            if S.assistant:
                assistant_response = S.assistant.handle(raw_text)
                if assistant_response is not None:
                    dur = time.time() - t0
                    entry = {"raw": raw_text, "cleaned": assistant_response, "app": app_name,
                             "duration": round(dur, 2), "ts": time.strftime("%H:%M:%S"),
                             "type": "assistant"}
                    add_history(entry)
                    emit({"type": "result", **entry})
                    _play_sound("done")
                    set_status("idle", f"Done ({dur:.1f}s)")
                    return

            # Fall through to text transform
            selected = get_selected_text()
            if selected:
                cleaned = S.cleaner.transform(selected, raw_text)
            else:
                cleaned = S.cleaner.clean(raw_text, app_name=app_name,
                                          context=extra_context,
                                          tone_override=S.tone_override,
                                          dictionary_terms=S.dictionary.terms if S.dictionary else [],
                                          style_prompt=style_prompt)
        else:
            cleaned = S.cleaner.clean(raw_text, app_name=app_name,
                                      context=extra_context,
                                      tone_override=S.tone_override,
                                      dictionary_terms=S.dictionary.terms if S.dictionary else [],
                                      style_prompt=style_prompt)

        if cleaned:
            if app_name not in BLOCKED_APPS:
                inject_text(cleaned, target_app=app_name)
            else:
                log.info("Skipped injection into '%s' — text saved to history", app_name)
            S.reset_after_paste(cleaned)
            if S.conversation:
                S.conversation.add_turn(app_name, cleaned)

        # === Compute diff ===
        diff = _compute_diff(raw_text, cleaned) if cleaned and cleaned != raw_text else []

        dur = time.time() - t0
        entry = {"raw": raw_text, "cleaned": cleaned or "", "app": app_name,
                 "duration": round(dur, 2), "ts": time.strftime("%H:%M:%S"),
                 "diff": diff}
        add_history(entry)
        emit({"type": "result", **entry})
        _play_sound("done")
        set_status("idle", f"Done ({dur:.1f}s)")

        # Auto-learn terms + detect deadlines (no auto memory — too noisy)
        if cleaned and len(cleaned.split()) >= 8:
            if S.dictionary and S.cleaner:
                threading.Thread(target=_auto_learn_terms, args=(cleaned,), daemon=True).start()
            if S.brain:
                dl = S.brain.detect_deadline_in_text(cleaned)
                if dl:
                    S.brain.add_deadline(dl["text"][:100], dl["due_hint"])
                    log.info("Deadline detected: %s", dl["due_hint"])

    except Exception as e:
        log.error("Processing error: %s", e, exc_info=True)
        msg = _friendly_error(e)
        _play_sound("error")
        emit({"type": "error", "error": msg})
        set_status("idle", msg)
    finally:
        S.processing = False


def _friendly_error(e: Exception) -> str:
    """Convert exceptions to user-friendly error messages."""
    err = str(e).lower()
    if "api_key" in err or "authentication" in err or "401" in err:
        return "Groq API key invalid or missing — check Settings"
    if "rate_limit" in err or "429" in err:
        return "Rate limited — wait a moment and try again"
    if "timeout" in err or "timed out" in err:
        return "Request timed out — check your connection"
    if "connection" in err or "network" in err or "unreachable" in err:
        return "Network error — check your internet connection"
    if "permission" in err or "not permitted" in err:
        return "Permission denied — check System Settings > Privacy"
    if "microphone" in err or "audio" in err or "sounddevice" in err:
        return "Microphone error — check mic access in System Settings"
    if "model" in err and ("not found" in err or "load" in err):
        return "Transcription model failed to load — try a different backend"
    if "out of memory" in err or "oom" in err:
        return "Out of memory — recording may be too long"
    return f"Unexpected error: {type(e).__name__}: {str(e)[:80]}"


def _auto_learn_terms(text: str):
    """Extract proper nouns, emails, acronyms from text and add to dictionary."""
    try:
        terms = S.cleaner.extract_terms(text)
        if not terms:
            return
        existing = set(t.lower() for t in S.dictionary.terms)
        added = []
        for term in terms:
            if term.lower() not in existing and len(term) > 1:
                S.dictionary.add_term(term)
                existing.add(term.lower())
                added.append(term)
        if added:
            log.info("Auto-learned: %s", ", ".join(added))
            emit({"type": "terms_learned", "terms": added})
    except Exception as e:
        log.debug("Auto-learn failed: %s", e)


def audio_level_monitor():
    """Broadcast audio RMS levels for the pill waveform animation."""
    import numpy as np
    smooth_rms = 0.0
    while True:
        if not (S.recorder and S.recorder.is_recording):
            smooth_rms = 0.0
            time.sleep(0.3)
            continue
        time.sleep(0.08)  # ~12fps for smooth animation
        try:
            with S.recorder._lock:
                if not S.recorder._frames:
                    continue
                n = min(3, len(S.recorder._frames))
                recent = np.concatenate([S.recorder._frames[-i].copy().flatten() for i in range(1, n + 1)])
            rms = float(np.sqrt(np.mean(recent ** 2)))
            smooth_rms = smooth_rms * 0.6 + rms * 0.4  # EMA smoothing
            emit({"type": "audio_level", "rms": round(smooth_rms, 4)})
        except Exception:
            pass


def request_mic_permission():
    """Request microphone permission on macOS."""
    try:
        import AVFoundation
        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
            AVFoundation.AVMediaTypeAudio
        )
        if status == 0:  # notDetermined
            log.info("Requesting mic permission...")
            import threading
            event = threading.Event()
            def handler(granted):
                log.info("Mic permission granted: %s", granted)
                event.set()
            AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVFoundation.AVMediaTypeAudio, handler
            )
            event.wait(timeout=5)  # don't block startup for long
        elif status == 2:  # denied
            log.warning("Mic permission DENIED — open System Settings > Privacy > Microphone")
        elif status == 3:  # authorized
            log.info("Mic permission already authorized")
    except Exception as e:
        log.warning("Could not check mic permission: %s", e)


def load_engine():
    # Check all permissions at startup
    try:
        import permissions
        print("\n  Checking permissions...")
        permissions.print_status()
    except Exception as e:
        log.warning("Permission check failed: %s", e)

    request_mic_permission()

    # Ensure Groq key is in Keychain (persists even if .env is lost)
    from config import GROQ_API_KEY, DATA_DIR, _keychain_get, _keychain_set
    if GROQ_API_KEY and not _keychain_get("Muse", "groq_api_key"):
        _keychain_set("Muse", "groq_api_key", GROQ_API_KEY)
        log.info("Saved Groq key to Keychain for persistence")

    # Copy .env to data dir if it doesn't exist there (packaged app support)
    data_env = DATA_DIR / ".env"
    if not data_env.exists() and GROQ_API_KEY:
        try:
            data_env.write_text(f"GROQ_API_KEY={GROQ_API_KEY}\n")
        except Exception:
            pass

    S.recorder = Recorder()
    # Restore persisted LLM mode + transcription backend
    _prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
    saved_llm = _prefs.get("llm_mode")
    if saved_llm in ("local", "groq", "hybrid"):
        os.environ["LLM_PROVIDER"] = saved_llm
    saved_backend = _prefs.get("transcription_backend",
                               "groq" if GROQ_API_KEY else "faster-whisper")
    S.transcriber = Transcriber(backend=saved_backend)
    S.cleaner = Cleaner()
    S.dictionary = PersonalDictionary()
    S.snippets = SnippetStore()
    S.styles = StyleManager()
    S.domains = DomainManager()
    S.macros = MacroEngine()
    S.conversation = ConversationTracker()
    S.oauth = OAuthManager()
    S.oauth.set_emit_fn(emit)
    S.todos = TodoList()
    S.brain = Brain()
    from llm import get_client as get_llm
    S.assistant = Assistant(get_llm(), S.oauth, emit, S.todos, S.brain)
    # Meeting system
    from meeting_recorder import MeetingRecorder
    from meeting_detector import MeetingDetector
    S.meeting_recorder = MeetingRecorder(S.transcriber, emit)
    S.meeting_detector = MeetingDetector(S.oauth, emit)
    # Meeting detector does NOT auto-start — user must enable it from Meetings page
    set_status("idle", "Ready — hold ⌥R to dictate")


def start_listener():
    # Check Input Monitoring permission
    try:
        import subprocess
        result = subprocess.run(
            ["/usr/local/bin/python3", "-c",
             "import ApplicationServices; print(ApplicationServices.AXIsProcessTrusted())"],
            capture_output=True, text=True, timeout=5
        )
        trusted = result.stdout.strip() == "True"
        if not trusted:
            log.warning("Input Monitoring may not be granted — hotkeys might not work")
            log.warning("Add Python and/or Muse.app to System Settings > Privacy > Input Monitoring")
    except Exception as e:
        log.debug("Could not check Input Monitoring: %s", e)

    log.info("Starting keyboard listener (dictation=⌥L, assistant=⌥R)")

    def on_press_safe(key):
        try:
            on_key_press(key)
        except Exception as e:
            log.error("Key press handler error: %s", e, exc_info=True)

    def on_release_safe(key):
        try:
            on_key_release(key)
        except Exception as e:
            log.error("Key release handler error: %s", e, exc_info=True)

    with keyboard.Listener(on_press=on_press_safe, on_release=on_release_safe) as listener:
        listener.join()


if __name__ == "__main__":
    # Check port BEFORE starting anything
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", PORT))
        sock.close()
    except OSError:
        print(f"\n  ERROR: Port {PORT} is already in use.", flush=True)
        print(f"  Kill the existing process: pkill -f 'app.py'", flush=True)
        sys.exit(1)

    threading.Thread(target=load_engine, daemon=True).start()
    threading.Thread(target=start_listener, daemon=True).start()
    threading.Thread(target=audio_level_monitor, daemon=True).start()

    # Follow-up checker (hourly)
    def follow_up_checker():
        import follow_ups
        while True:
            time.sleep(3600)
            try:
                reminders = follow_ups.get_pending_reminders()
                for r in reminders:
                    msg = f"No reply from {r['to']} about '{r['subject']}' ({r['days_ago']}d ago)"
                    pill_notify(msg, "followup")
                    emit({"type": "follow_up_reminder", "reminders": reminders})
                    follow_ups.mark_reminded(r["to"], r["subject"])
                    log.info("Follow-up reminder: %s", msg)
            except Exception as e:
                log.debug("Follow-up check: %s", e)
    threading.Thread(target=follow_up_checker, daemon=True).start()

    # Periodic memory queue flusher
    def memory_flusher():
        while True:
            time.sleep(30)
            try:
                mem.flush_pending()
            except Exception:
                pass
    threading.Thread(target=memory_flusher, daemon=True).start()

    # Meeting notifier — pill_notify when a meeting is 5 min away
    def meeting_notifier():
        _notified = set()
        while True:
            time.sleep(60)
            try:
                token = S.oauth.get_token("google") if S.oauth else None
                if not token:
                    continue
                from integrations.google_calendar import list_events
                from datetime import datetime as dt, timezone as tz
                r = list_events(token, days_ahead=1, max_results=10)
                for ev in r.get("events", []):
                    eid = ev.get("id", "")
                    if eid in _notified:
                        continue
                    try:
                        start = dt.fromisoformat(ev["start"].replace("Z", "+00:00"))
                        diff = (start - dt.now(tz.utc)).total_seconds()
                        if 240 <= diff <= 360:
                            _notified.add(eid)
                            pill_notify(f"Meeting in 5 min: {ev.get('summary','')[:40]}", "calendar", 5000)
                    except (ValueError, KeyError):
                        pass
            except Exception:
                pass
    threading.Thread(target=meeting_notifier, daemon=True).start()

    # Reply watcher — checks for iMessage replies every 15s
    def reply_watcher():
        import imessage
        while True:
            time.sleep(15)
            try:
                replies = imessage.check_for_replies()
                for r in replies:
                    msg = f"{r['contact']} replied: \"{r['text']}\""
                    emit({"type": "assistant_stream", "text": msg, "done": True})
                    emit({"type": "messages_view", "messages": [
                        {"from": r["contact"], "text": r["text"], "time": r["time"]}
                    ]})
                    # Speak the notification
                    try:
                        tts.speak(f"{r['contact']} responded. They said: {r['text']}")
                    except Exception:
                        pass
                    log.info("Reply notification: %s", msg)
            except Exception as e:
                log.debug("Reply watcher: %s", e)
    threading.Thread(target=reply_watcher, daemon=True).start()

    if "--no-browser" not in sys.argv:
        def open_browser():
            time.sleep(2)
            webbrowser.open(f"http://localhost:{PORT}")
        threading.Thread(target=open_browser, daemon=True).start()

    import signal
    def shutdown(sig, frame):
        print("\n  Shutting down...", flush=True)
        if S.recorder and S.recorder.is_recording:
            S.recorder.stop()
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"\n  Muse → http://localhost:{PORT}\n", flush=True)
    uvicorn.run(api, host="127.0.0.1", port=PORT, log_level="warning")
