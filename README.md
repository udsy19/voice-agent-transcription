# Muse

Voice dictation + AI assistant for macOS. Hold a key, speak, release — cleaned text appears wherever your cursor is. Or talk to the assistant to manage your calendar, email, todos, and memories.

## Features

**Dictation (⌥L)**
- Hold Left Option, speak, release — text is transcribed, cleaned, and pasted at your cursor
- Filler word removal, grammar fixes, punctuation
- App-aware tone (formal in Docs, casual in Slack, code in VS Code)
- Auto language detection (99 languages)
- Double-tap ⌥L for toggle mode (keeps listening until you press again)

**AI Assistant (⌥R)**
- Hold Right Option to talk to the assistant
- Multi-turn conversations — it asks follow-up questions if info is missing
- Speaks responses aloud via Kokoro TTS (local, 12 voices)

**Calendar**
- "What's on my calendar today?" — lists events with times
- "Add a meeting tomorrow at 2pm with John" — creates events
- "Add a Google Meet link" — generates video call link
- Attendees, location, notes, timezone support
- Update and delete events by voice
- Conflict detection before creating
- Mini calendar view in the pill overlay

**Email**
- "Draft an email to john@example.com about the project" — creates Gmail draft
- "Send it" — sends the draft
- Defaults to drafting for safety

**Todos**
- "Add buy groceries to my todo list" — adds task
- "I've bought groceries, check it off" — completes task
- Todo widget on home page with inline add

**Memory (Mem0)**
- "Remember that Samyukta is my girlfriend" — stored permanently
- "What's my favorite coffee shop?" — recalls from memory
- "It's not Anisha, it's Tanisha" — corrects automatically (delete + add)
- "Forget Anisha Mittal" — deletes memories
- Categorized memory page (People, Preferences, Travel, Work, School, Tech)
- Local vector DB (Qdrant) + HuggingFace embeddings — no cloud dependency

**Quick Capture (dictation mode)**
- "Remind me to buy groceries" → adds todo
- "Note: API rate limit is 100" → saves to memory
- "Meeting notes: discussed Q2 roadmap" → saves meeting notes

## Architecture

```
┌─ Electron (.app) ──────────────────────────────┐
│  Pill overlay (waveform + assistant text)       │
│  Settings UI (Notion-inspired dark theme)       │
└────────────── WebSocket ───────────────────────┘
                    │
┌─ Python Backend (FastAPI :8528) ───────────────┐
│  recorder.py      — mic capture + resampling   │
│  transcriber.py   — Groq/faster-whisper/MLX    │
│  cleaner.py       — LLM text cleanup           │
│  assistant.py     — tool-calling AI agent      │
│  memory.py        — Mem0 vector memory         │
│  tts.py           — Kokoro ONNX local TTS      │
│  injector.py      — paste into active app      │
│  todos.py         — persistent todo list       │
│  brain.py         — deadlines, meeting notes   │
│  integrations/    — Google Calendar + Gmail     │
└─────────────────────────────────────────────────┘
```

## Online vs Offline

| Function | Online | Offline |
|----------|--------|---------|
| Transcription | Groq whisper-large-v3 | faster-whisper (local CPU) |
| Text cleanup | Groq llama-4-scout-17b | Local regex only |
| Assistant | Groq llama-3.3-70b | Not available |
| TTS | Kokoro ONNX (always local) | Same |
| Memory search | Qdrant (always local) | Same |
| Embeddings | HuggingFace (always local) | Same |

## Setup

### Prerequisites
- macOS (Apple Silicon or Intel)
- Python 3.13+
- Node.js 20+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Install

```bash
git clone https://github.com/udsy19/voice-agent-transcription.git
cd voice-agent-transcription

# Python dependencies
pip install -r requirements.txt

# Electron dependencies
cd electron && npm install && cd ..

# Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### Run

```bash
./start.sh
```

This starts the Python backend + Electron UI. The app appears in your menu bar tray.

### Permissions

You need to grant these in System Settings > Privacy & Security:

1. **Microphone** — for voice recording
2. **Input Monitoring** — for hotkey detection (add Python.app and Terminal.app)
3. **Accessibility** — for text injection via paste

### Google Calendar + Gmail (optional)

1. Go to [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials)
2. Create OAuth Client ID (type: Desktop app)
3. Enable Google Calendar API + Gmail API
4. In Muse: Settings > Integrations > enter Client ID + Secret > Connect

### Kokoro TTS (optional, auto-downloads)

On first use, Kokoro downloads ~337MB of model files to `~/Library/Application Support/Muse/models/`. This happens automatically.

## Packaged App

```bash
cd electron
npm run build        # builds .app to electron/dist/mac-arm64/Muse.app
```

## Data Storage

| Data | Location |
|------|----------|
| Preferences | `~/Library/Application Support/Muse/preferences.json` |
| Todos | `~/Library/Application Support/Muse/todos.json` |
| Memory DB | `~/Library/Application Support/Muse/memory_db/` |
| Dictionary | `~/Library/Application Support/Muse/personal_dictionary.json` |
| TTS models | `~/Library/Application Support/Muse/models/` |
| Logs | `~/Library/Logs/Muse/muse.log` |
| API keys | macOS Keychain (service: "Muse") |

## Tech Stack

- **Backend**: Python 3.13, FastAPI, uvicorn
- **Frontend**: Electron, vanilla HTML/CSS/JS
- **Transcription**: Groq API (whisper-large-v3), faster-whisper, MLX Whisper
- **LLM**: Groq API (llama-3.3-70b-versatile, llama-4-scout-17b, llama-3.1-8b-instant)
- **TTS**: Kokoro ONNX (local, 12 voices)
- **Memory**: Mem0 + Qdrant (local vector DB) + HuggingFace sentence-transformers
- **Calendar/Email**: Google Calendar API, Gmail API
- **Key storage**: macOS Keychain
- **Hotkeys**: pynput
- **Audio**: sounddevice, scipy (resampling)

## License

MIT
