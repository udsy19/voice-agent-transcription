# Voice Agent — CLAUDE.md

## What This Is
A local voice dictation tool (like Wispr Flow) that works system-wide on macOS. Hold a hotkey, speak, release — cleaned text appears wherever your cursor is.

## Architecture
- **Python backend** — faster-whisper (local ASR) + Groq API (LLM cleanup) + pynput (hotkeys) + Quartz (text injection)
- **Electron frontend** — Claura design system, shows floating pill when recording, settings/history window
- Backend and frontend communicate via local WebSocket on localhost:8528

## Key Decisions

### Terminal & Permissions
- **cmux does NOT have mic access** — the Python backend must be launched from /usr/bin/open or from System Terminal, NOT from cmux
- Permissions needed: Microphone, Input Monitoring, Accessibility
- The Electron app itself should request these permissions (it's a signed .app so macOS will prompt)

### Hotkey
- **Right Option key (⌥R)** — hold to record, release to transcribe
- This is a HOLD-to-record model, not tap-to-start/tap-to-stop
- pynput on_press starts recording, on_release stops and processes

### Text Injection
- Text gets pasted wherever the cursor was BEFORE the Voice Agent window was opened
- Uses pbcopy + Quartz CGEvent Cmd+V (most reliable cross-app method)
- After pasting, the cleaned text also stays on the clipboard so user can Cmd+V again
- If paste fails, the text goes to history and user can retrieve it from there

### Floating Pill UI
- When recording: show a small floating pill/overlay (like Wispr Flow's wave indicator)
- When processing: pill shows loading/status animation
- When done: pill disappears, text is pasted
- This is NOT a menubar icon — it's a floating overlay window always-on-top

### App Behavior
- Runs in background — closing the window does NOT quit the app
- Electron app should have a tray/dock presence
- Settings window accessible from tray menu or hotkey
- Auto-start on login (optional, user-toggleable)
- Cross-platform target: macOS first, Windows later

### Design System
- **Claura design system** (from /Users/udsy/Downloads/Claude Design System.skill)
- Colors: bg #FAF6F0, primary #1C0E04, accent #ECE3D8, amber #CC8A24, muted #877263
- Fonts: Halant (headings/serif), DM Sans (body/UI)
- Warm browns, no pure black/white, rounded corners (12-14px)
- Shadows: warm brown tones, never cool grey

### Voice Engine
- **Model**: distil-large-v3 (faster-whisper, runs locally on CPU)
- **Language**: English only for now (hardcode language="en")
- **Compute**: auto (let ctranslate2 pick best for hardware)
- **Speed priority**: beam_size=1, best_of=1, condition_on_previous_text=False
- **Cleanup**: Groq API with llama-3.1-8b-instant (fastest model)
- **Groq API key**: stored in .env (never commit this file)

### Features (Wispr Flow parity)
1. Hold-to-record dictation → paste anywhere
2. Filler word removal, grammar fixes, punctuation
3. Backtracking ("Tuesday... actually Wednesday" → "Wednesday")
4. List formatting (first/second/third → numbered list)
5. App-aware tone (formal in Docs, casual in Slack, code in VS Code)
6. Personal dictionary (terms + auto-corrections)
7. Voice snippets (trigger phrase → text block)
8. Command mode ("Hey Flow, make this more professional" transforms selected text)
9. History page with all dictations
10. Developer/code mode in editors

### What NOT to Do
- Don't use rumps for menubar — it doesn't show up
- Don't try native Swift .app — Gatekeeper blocks unsigned apps silently
- Don't use cmux to launch the Python backend — no mic access
- Don't use large-v3 model — too slow on CPU, use distil-large-v3
- Don't use float16 compute type — ctranslate2 on CPU doesn't support it

## File Structure
```
voice-agent/
├── app.py              # FastAPI backend (voice engine + API)
├── backend.py          # Standalone backend (for non-web use)
├── main.py             # Terminal-only entry point
├── config.py           # All settings
├── recorder.py         # Mic capture + whisper mode
├── transcriber.py      # faster-whisper wrapper
├── cleaner.py          # Groq LLM cleanup + tone adaptation
├── injector.py         # Quartz CGEvent text injection
├── dictionary.py       # Personal dictionary
├── snippets.py         # Voice snippets
├── styles.py           # Tone profiles
├── sync.py             # Settings sync (iCloud/Dropbox)
├── logger.py           # File-based logging
├── electron/           # Electron desktop app
│   ├── main.js         # Electron main process
│   ├── index.html      # UI (Claura design)
│   └── package.json
├── ui/                 # Web UI (fallback)
│   └── index.html
├── .env                # GROQ_API_KEY
├── requirements.txt
├── personal_dictionary.json
├── snippets.json
└── styles.json
```

## Running
```bash
# Install deps
pip install -r requirements.txt

# Terminal mode (simplest)
python3 main.py

# Web dashboard mode
python3 app.py

# Electron app (TODO: rebuild properly)
cd electron && npm start
```

## Environment
- macOS 26.3.1, Apple M4 Pro
- Python 3.13 at /usr/local/bin/python3
- Node 24.9.0 at /opt/homebrew/bin/node
- Mic native rate: 48000 Hz (must resample to 16000 Hz for Whisper)
- Uses scipy.signal.resample_poly for high-quality resampling
