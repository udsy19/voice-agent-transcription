#!/usr/bin/env python3
"""Voice Agent — web dashboard + voice backend.

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
import uvicorn

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
from logger import get_logger

log = get_logger("app")
PORT = 8528
TRIGGER_KEY = keyboard.Key.alt_r
HANDS_FREE_SILENCE_SEC = 2.0

# Undo keywords
UNDO_PHRASES = {"undo that", "go back", "undo", "scratch that", "never mind", "cancel that"}


# ── Global State ─────────────────────────────────────────────────────────────

class State:
    recorder: Recorder = None
    transcriber: Transcriber = None
    cleaner: Cleaner = None
    dictionary: PersonalDictionary = None
    snippets: SnippetStore = None
    styles: StyleManager = None
    domains: DomainManager = None
    macros: MacroEngine = None

    status = "loading"
    detail = "Loading model..."
    processing = False
    hands_free = False
    whisper_mode = False
    history: list = []  # capped at 100 entries
    ws_clients: list = []
    key_held = False
    last_paste_text = ""
    last_paste_time = 0.0
    tone_override = None  # set by macros

S = State()


async def broadcast(msg: dict):
    dead = []
    for ws in S.ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        S.ws_clients.remove(ws)


def emit(msg: dict):
    loop = getattr(emit, '_loop', None)
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(msg), loop)


def add_history(entry):
    add_history(entry)
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
async def api_set_domain(body: dict):
    d = body.get("domain", "").strip()
    if S.domains:
        S.domains.set_active(d)
    return {"ok": True}


@api.post("/api/macros/add")
async def api_add_macro(body: dict):
    trigger = body.get("trigger", "").strip()
    desc = body.get("description", "").strip()
    actions = body.get("actions", [])
    if trigger and actions and S.macros:
        S.macros.add(trigger, desc, actions)
    return {"ok": True}


@api.post("/api/macros/remove")
async def api_remove_macro(body: dict):
    trigger = body.get("trigger", "").strip()
    if trigger and S.macros:
        S.macros.remove(trigger)
    return {"ok": True}


@api.post("/api/dictionary/add-term")
async def api_add_term(body: dict):
    t = body.get("term", "").strip()
    if t and S.dictionary:
        S.dictionary.add_term(t)
    return {"ok": True}


@api.post("/api/dictionary/add-correction")
async def api_add_correction(body: dict):
    w, c = body.get("wrong", "").strip(), body.get("correct", "").strip()
    if w and c and S.dictionary:
        S.dictionary.add_correction(w, c)
    return {"ok": True}


@api.post("/api/dictionary/remove-term")
async def api_remove_term(body: dict):
    t = body.get("term", "").strip()
    if t and S.dictionary:
        S.dictionary.remove_term(t)
    return {"ok": True}


@api.post("/api/snippets/add")
async def api_add_snippet(body: dict):
    t, txt = body.get("trigger", "").strip(), body.get("text", "").strip()
    if t and txt and S.snippets:
        S.snippets.add(t, txt)
    return {"ok": True}


@api.post("/api/snippets/remove")
async def api_remove_snippet(body: dict):
    t = body.get("trigger", "").strip()
    if t and S.snippets:
        S.snippets.remove(t)
    return {"ok": True}


@api.post("/api/set-groq-key")
async def api_set_groq_key(body: dict):
    """Set Groq API key from the settings UI. Saves to .env and reinitializes cleaner."""
    key = body.get("key", "").strip()
    if not key:
        return {"ok": False, "reason": "empty key"}
    # Save to .env
    from pathlib import Path
    from config import DATA_DIR
    env_path = DATA_DIR / ".env"
    env_path.write_text(f"GROQ_API_KEY={key}\n")
    # Also save to project dir for dev mode
    try:
        (Path(__file__).parent / ".env").write_text(f"GROQ_API_KEY={key}\n")
    except Exception:
        pass
    # Reinitialize cleaner
    import config
    config.GROQ_API_KEY = key
    from groq import Groq
    S.cleaner._client = Groq(api_key=key)
    log.info("Groq API key updated")
    return {"ok": True}


@api.get("/api/groq-status")
async def api_groq_status():
    has_key = bool(S.cleaner and S.cleaner._client)
    return {"configured": has_key}


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
    result["groq_key"] = bool(S.cleaner and S.cleaner._client)

    return result


@api.get("/api/transcription-backend")
async def api_get_backend():
    backend = S.transcriber.backend if S.transcriber else "loading"
    return {"backend": backend, "available": ["parakeet", "mlx", "groq", "faster-whisper"]}


@api.post("/api/transcription-backend")
async def api_set_backend(body: dict):
    backend = body.get("backend", "").strip()
    if backend and S.transcriber:
        S.transcriber.set_backend(backend)
        return {"ok": True, "backend": S.transcriber.backend}
    return {"ok": False}


@api.post("/api/styles/set-role")
async def api_set_role(body: dict):
    r = body.get("role", "").strip()
    if r and S.styles:
        S.styles.setup_role(r)
    return {"ok": True}


@api.post("/api/toggle-hands-free")
async def api_toggle_hf():
    toggle_hands_free()
    return {"ok": True}


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
    except WebSocketDisconnect:
        if ws in S.ws_clients:
            S.ws_clients.remove(ws)


# ── Voice Engine ─────────────────────────────────────────────────────────────

_key_lock = threading.Lock()


def on_key_press(key):
    """Hold Right Option to start recording."""
    if key != TRIGGER_KEY:
        return
    with _key_lock:
        if S.key_held or S.processing or S.hands_free:
            return
        if not S.recorder:
            return
        S.key_held = True
        if not S.recorder.is_recording:
            S.recorder.start()
            set_status("recording", "Listening — release to transcribe")


def on_key_release(key):
    if key != TRIGGER_KEY:
        return
    with _key_lock:
        if S.hands_free:
            toggle_hands_free()
            return
        if S.key_held:
            S.key_held = False
            if S.recorder and S.recorder.is_recording:
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
        if not S.recorder._frames:
            continue
        with S.recorder._lock:
            if not S.recorder._frames:
                continue
            recent = S.recorder._frames[-1].flatten()
        rms = np.sqrt(np.mean(recent ** 2))
        threshold = SILENCE_THRESHOLD * 0.5 if S.whisper_mode else SILENCE_THRESHOLD
        if rms < threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= HANDS_FREE_SILENCE_SEC:
                if len(S.recorder._frames) > 5:
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
    audio = S.recorder.stop()
    if audio is None:
        set_status("idle", "Too short or silent")
        return
    S.processing = True
    threading.Thread(target=process_audio, args=(audio,), daemon=True).start()



def _get_clipboard_context() -> str:
    """Read current clipboard for context injection into Groq."""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2)
        text = result.stdout.strip()
        # Only use if it looks like real text (not binary/huge)
        if text and len(text) < 2000 and text.isprintable():
            return text[:500]
    except Exception:
        pass
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
        app_name = _get_active_app()
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
                inject_text(snippet_text)
                S.last_paste_text = snippet_text
                S.last_paste_time = time.time()
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
        clipboard_context = _get_clipboard_context()

        # === Get domain hint ===
        domain_hint = S.domains.get_cleaner_hint() if S.domains else None

        # === Build extra context for cleaner ===
        extra_context = ""
        if clipboard_context:
            extra_context += clipboard_context
        if domain_hint:
            extra_context += f"\n{domain_hint}" if extra_context else domain_hint

        if is_command:
            selected = get_selected_text()
            if selected:
                cleaned = S.cleaner.transform(selected, raw_text)
            else:
                cleaned = S.cleaner.clean(raw_text, app_name=app_name,
                                          context=extra_context,
                                          tone_override=S.tone_override,
                                          dictionary_terms=S.dictionary.terms if S.dictionary else [])
        else:
            cleaned = S.cleaner.clean(raw_text, app_name=app_name,
                                      context=extra_context,
                                      tone_override=S.tone_override,
                                      dictionary_terms=S.dictionary.terms if S.dictionary else [])

        # Reset tone override after use (macros set it temporarily)
        S.tone_override = None

        if cleaned:
            inject_text(cleaned)
            S.last_paste_text = cleaned
            S.last_paste_time = time.time()

        # === Compute diff ===
        diff = _compute_diff(raw_text, cleaned) if cleaned and cleaned != raw_text else []

        dur = time.time() - t0
        entry = {"raw": raw_text, "cleaned": cleaned or "", "app": app_name,
                 "duration": round(dur, 2), "ts": time.strftime("%H:%M:%S"),
                 "diff": diff}
        add_history(entry)
        emit({"type": "result", **entry})
        set_status("idle", f"Done ({dur:.1f}s)")

        # Auto-learn: extract notable terms in background
        if cleaned and S.dictionary and S.cleaner:
            threading.Thread(target=_auto_learn_terms, args=(cleaned,), daemon=True).start()

    except Exception as e:
        log.error("Error: %s", e, exc_info=True)
        set_status("idle", f"Error: {e}")
    finally:
        S.processing = False


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
    while True:
        if not (S.recorder and S.recorder.is_recording):
            time.sleep(0.5)
            continue
        time.sleep(0.15)
        if S.recorder._frames:
            try:
                with S.recorder._lock:
                    if S.recorder._frames:
                        recent = S.recorder._frames[-1].flatten()
                        rms = float(np.sqrt(np.mean(recent ** 2)))
                        quality = "good" if rms > 0.03 else "fair" if rms > 0.01 else "low"
                        emit({"type": "audio_level", "rms": rms, "quality": quality})
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
            event.wait(timeout=30)
        elif status == 2:  # denied
            log.warning("Mic permission DENIED — open System Settings > Privacy > Microphone")
        elif status == 3:  # authorized
            log.info("Mic permission already authorized")
    except Exception as e:
        log.warning("Could not check mic permission: %s", e)


def load_engine():
    request_mic_permission()
    S.recorder = Recorder()
    S.transcriber = Transcriber(backend="parakeet")  # fastest local, falls back automatically
    S.cleaner = Cleaner()
    S.dictionary = PersonalDictionary()
    S.snippets = SnippetStore()
    S.styles = StyleManager()
    S.domains = DomainManager()
    S.macros = MacroEngine()
    set_status("idle", "Ready — hold ⌥R to dictate")


def start_listener():
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

    print(f"\n  Voice Agent → http://localhost:{PORT}\n", flush=True)
    uvicorn.run(api, host="127.0.0.1", port=PORT, log_level="warning")
