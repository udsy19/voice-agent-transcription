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
from config import SILENCE_THRESHOLD
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
from exporter import export_json, export_txt, export_word, export_pdf, export_logs
from assistant import Assistant
from integrations.oauth_manager import OAuthManager
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
_sound_enabled = True
_clipboard_context_enabled = True

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
    for ws in S.ws_clients:
        try:
            await ws.send_json(msg)
        except Exception as e:
            log.debug("WebSocket send failed: %s", e)
            dead.append(ws)
    for ws in dead:
        if ws in S.ws_clients:
            S.ws_clients.remove(ws)


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
    allow_origins=["http://localhost:*", "http://127.0.0.1:*", "file://"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
        "snippets": {k: v for k, v in ((k, v["text"]) for k, v in S.snippets._snippets.items())} if S.snippets else {},
        "role": S.styles._user_role if S.styles else "",
        "default_style": S.styles._default_style if S.styles else "",
        "app_overrides": S.styles._app_overrides if S.styles else {},
        "domains": S.domains.list_domains() if S.domains else {},
        "active_domain": S.domains.get_active() if S.domains else "",
        "transcription_backend": S.transcriber.backend if S.transcriber else "loading",
        "macros": S.macros.list_all() if S.macros else {},
    }


@api.post("/api/domains/set")
async def api_set_domain(body: DomainRequest):
    d = body.domain.strip()
    if S.domains:
        S.domains.set_active(d)
    return {"ok": True}


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
    has_client = bool(S.cleaner and S.cleaner._client)
    # Mask key for display (show first 8 chars)
    masked = GROQ_API_KEY[:4] + "***" + GROQ_API_KEY[-4:] if GROQ_API_KEY and len(GROQ_API_KEY) > 12 else "***" if GROQ_API_KEY else ""
    return {"configured": has_key or has_client, "masked_key": masked}


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
        return {"ok": False, "reason": f"export failed: {e}"}


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


@api.delete("/api/accounts/{service}/{email}")
async def api_remove_account(service: str, email: str):
    """Disconnect an OAuth account."""
    if S.oauth:
        S.oauth.remove_account(service, email)
    return {"ok": True}


@api.get("/api/permissions")
async def api_permissions():
    """Check microphone and input monitoring permissions."""
    import sys
    result = {"mic": "unknown", "input_monitoring": "unknown", "python_path": sys.executable}

    # Check mic permission
    try:
        import AVFoundation
        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
            AVFoundation.AVMediaTypeAudio
        )
        result["mic"] = {0: "not_asked", 1: "restricted", 2: "denied", 3: "granted"}.get(status, "unknown")
    except Exception:
        result["mic"] = "unknown"

    # Check input monitoring — try a quick test recording
    try:
        import sounddevice as sd
        import numpy as np
        audio = sd.rec(int(0.1 * 48000), samplerate=48000, channels=1, dtype='float32')
        sd.wait()
        rms = float(np.sqrt(np.mean(audio ** 2)))
        result["mic_working"] = rms > 0.0001
    except Exception:
        result["mic_working"] = False

    # Check if pynput can see key events (we check if the listener printed the warning)
    # We can't test this directly, so we check if the process is "trusted"
    try:
        import Quartz
        trusted = Quartz.AXIsProcessTrusted()
        result["input_monitoring"] = "granted" if trusted else "denied"
    except Exception:
        result["input_monitoring"] = "unknown"

    # Groq API key
    from config import GROQ_API_KEY
    result["groq_key"] = bool(GROQ_API_KEY) or bool(S.cleaner and S.cleaner._client)

    return result


@api.get("/api/transcription-backend")
async def api_get_backend():
    backend = S.transcriber.backend if S.transcriber else "loading"
    return {"backend": backend, "available": ["parakeet", "mlx", "groq", "faster-whisper"]}


@api.post("/api/transcription-backend")
async def api_set_backend(body: BackendRequest):
    backend = body.backend.strip()
    if backend and S.transcriber:
        S.transcriber.set_backend(backend)
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
    import json
    from config import DATA_DIR
    hk_path = os.path.join(str(DATA_DIR), "hotkey.json")
    try:
        with open(hk_path, "w") as f:
            json.dump({"key": key_name}, f)
    except Exception as e:
        log.error("Failed to save hotkey: %s", e)
    log.info("Hotkey changed to: %s", key_name)
    return {"ok": True, "key": key_name}


@api.post("/api/toggle-sounds")
async def api_toggle_sounds():
    global _sound_enabled
    _sound_enabled = not _sound_enabled
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
        # Safety: reset stuck key_held after 30s
        if S.key_held and (time.time() - _key_held_since) > 30:
            S.key_held = False

        if S.processing:
            return

        # If in toggle mode and user presses ⌥L again → stop
        if mode == "dictation" and _dictation_toggle and S.recorder and S.recorder.is_recording:
            _dictation_toggle = False
            stop_and_process()
            return

        if S.key_held or S.hands_free:
            return
        if not S.recorder:
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
        return
    if audio is None:
        set_status("idle", "Too short or silent")
        return
    S.processing = True

    def process_with_timeout():
        try:
            process_audio(audio)
        finally:
            S.processing = False  # guaranteed reset

    threading.Thread(target=process_with_timeout, daemon=True).start()



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

        # Auto-learn: extract notable terms in background (only for substantial text)
        if cleaned and len(cleaned.split()) >= 8 and S.dictionary and S.cleaner:
            threading.Thread(target=_auto_learn_terms, args=(cleaned,), daemon=True).start()

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
    request_mic_permission()
    S.recorder = Recorder()
    # Default to Groq API (most accurate + fast), fall back to faster-whisper if no key
    from config import GROQ_API_KEY
    default_backend = "groq" if GROQ_API_KEY else "faster-whisper"
    S.transcriber = Transcriber(backend=default_backend)
    S.cleaner = Cleaner()
    S.dictionary = PersonalDictionary()
    S.snippets = SnippetStore()
    S.styles = StyleManager()
    S.domains = DomainManager()
    S.macros = MacroEngine()
    S.conversation = ConversationTracker()
    S.oauth = OAuthManager()
    S.assistant = Assistant(S.cleaner._client, S.oauth, emit)
    set_status("idle", "Ready — hold ⌥R to dictate")


def start_listener():
    try:
        import Quartz
        if not Quartz.AXIsProcessTrusted():
            log.error("Input Monitoring not granted — hotkeys won't work")
            set_status("error", "Grant Input Monitoring in System Settings > Privacy")
    except Exception:
        pass
    with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
        listener.join()


if __name__ == "__main__":
    # Check port BEFORE starting anything
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
