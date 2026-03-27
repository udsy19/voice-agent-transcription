# Voice Agent

Local voice dictation + AI desktop assistant. Hold a key, speak, release — text appears at your cursor. Or switch to conversation mode and control your Mac with voice.

![Hold to dictate](https://img.shields.io/badge/Hold_⌥R-Dictate-1C0E04?style=for-the-badge)
![Conversation](https://img.shields.io/badge/Right_Click_Pill-Conversation-4040B4?style=for-the-badge)
![Local First](https://img.shields.io/badge/Privacy-Local_First-38A169?style=for-the-badge)

---

## Two Modes

### Dictation Mode (default)
Hold Right Option → speak → release → cleaned text pastes at cursor. Works in any app.

### Conversation Mode (right-click pill to toggle)
Hold Right Option → speak → release → agent responds via voice. Can see your screen, open apps, read messages, manage calendar, control your Mac.

---

## Features

**Dictation**
- Hold-to-record with global hotkey (Right Option)
- Multi-backend transcription: Groq API (fastest), MLX Whisper (GPU), Parakeet (local), faster-whisper (CPU)
- Hybrid AI cleanup: local filler removal + Groq LLM for complex cases
- Voice isolation (DeepFilterNet3) — removes background noise
- Smart undo, backtracking, list formatting, command mode
- Auto-learning dictionary, voice snippets, voice macros
- Per-app tone adaptation (formal in Docs, casual in Slack, code in VS Code)
- Diff view showing what was changed

**Conversation**
- 3-tier intent routing: instant (<100ms) → Groq (~400ms) → Claude (~800ms)
- Desktop control: AppleScript + Accessibility API + Terminator
- Screen awareness: screenshots + active app context sent to Claude
- Tool execution: open apps, read messages/emails, type text, click elements
- Persistent memory: session history + LanceDB vector store + knowledge graph
- User profile that evolves across sessions
- Kokoro TTS with emotional speed (excited=faster, serious=slower)
- Spoken narration at every step ("Opening Messages..." → "Found 3 conversations...")

**Desktop App**
- Electron app with floating pill (idle capsule → recording waveform → processing → done)
- Sidebar UI: Home, Dictionary, Snippets, Style, Macros, Scratchpad, Settings
- Setup wizard with permission checks (mic, input monitoring, API key)
- Configurable transcription engine from Settings dropdown
- Persistent data in `~/Library/Application Support/VoiceAgent/`

---

## Quick Start

### Prerequisites

- **macOS** (Apple Silicon or Intel)
- **Python 3.11+**
- **Node.js 18+**
- **Groq API key** — free at [console.groq.com](https://console.groq.com)
- **Anthropic API key** (for conversation mode) — [console.anthropic.com](https://console.anthropic.com)

### Install

```bash
git clone https://github.com/udsy19/voice-agent-transcription.git
cd voice-agent-transcription

# Python dependencies
pip install -r requirements.txt

# Electron dependencies
cd electron && npm install && cd ..

# Set API keys
cp .env.example .env
# Edit .env with your keys
```

### Run

```bash
# From Terminal.app (required for mic access)
bash start.sh
```

This starts the Python backend + Electron UI. The floating pill appears at the top of your screen.

**First time:** macOS will prompt for Microphone, Input Monitoring, and Accessibility permissions. The app shows a setup wizard to guide you.

### Permissions

| Permission | Why | Where to grant |
|---|---|---|
| Microphone | Record voice | System Settings → Privacy → Microphone |
| Input Monitoring | Global hotkey | System Settings → Privacy → Input Monitoring → add Python.app |
| Accessibility | Paste text | System Settings → Privacy → Accessibility |

---

## Architecture

```
┌─ Electron ──────────────────────────────────────────────┐
│  Main window (settings)  │  Floating pill (waveform)    │
│  Tray icon               │  Tooltip: "Hold ⌥R"         │
└──────────────────────────┼──────────────────────────────┘
                           │ WebSocket + REST
┌─ Python Backend ─────────┼──────────────────────────────┐
│                          │                              │
│  DICTATION MODE          │  CONVERSATION MODE           │
│  recorder → transcriber  │  recorder → transcriber      │
│  → cleaner → injector    │  → router → Claude/Groq      │
│  (paste at cursor)       │  → tools → TTS (speak back)  │
│                          │                              │
│  Shared: dictionary, snippets, macros, styles, domains  │
│  Persistent: ~/Library/Application Support/VoiceAgent/  │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
├── app.py                  # FastAPI server + voice engine + conversation endpoints
├── recorder.py             # Mic capture, silence detection, voice isolation
├── transcriber.py          # Multi-backend: Groq API, MLX, Parakeet, faster-whisper
├── cleaner.py              # Hybrid cleanup: local regex + Groq LLM
├── injector.py             # Text injection: clipboard paste + char-by-char typing
├── dictionary.py           # Personal dictionary (terms + auto-corrections)
├── snippets.py             # Voice-triggered text snippets
├── macros.py               # Chainable voice workflows
├── domains.py              # Domain vocabulary (tech, medical, legal, finance)
├── styles.py               # Tone profiles and per-app overrides
├── config.py               # All configuration (env var overrides)
├── logger.py               # Rotating file logger
├── start.sh                # Launcher script
│
├── conversation/           # Conversation mode
│   ├── agent.py            # Main loop: route → LLM → tools → TTS
│   ├── router.py           # 3-tier intent routing
│   ├── executor.py         # Structured tool execution
│   ├── desktop.py          # AppleScript + Accessibility + Terminator
│   ├── tools.py            # Tool definitions for Claude API
│   ├── context.py          # Screen capture + active app
│   ├── tts.py              # Kokoro TTS (streaming, emotional)
│   ├── memory.py           # Session + LanceDB + knowledge graph + user profile
│   ├── metrics.py          # Latency/error tracking
│   ├── mcp_client.py       # MCP server connections
│   └── vad_stream.py       # Silero VAD (continuous listening)
│
├── electron/
│   ├── main.js             # Electron main process
│   ├── preload.js          # Context bridge
│   └── ui/
│       ├── app.html        # Settings window (Claura design)
│       └── pill.html       # Floating pill overlay
│
└── .env                    # API keys (not committed)
```

---

## Voice Commands

### Dictation Mode

| Say this | What happens |
|---|---|
| *(hold ⌥R and speak)* | Dictates and pastes text |
| "undo that" / "go back" | Reverts the last paste |
| "Hey Flow, make this more professional" | Transforms selected text |
| "email mode" / "code mode" | Activates voice macro |
| "insert *(trigger phrase)*" | Pastes matching snippet |
| "scratch that" | Deletes preceding sentence |
| "new paragraph" | Inserts line break |

### Conversation Mode

| Say this | What happens |
|---|---|
| "open Slack" | Opens the app (~80ms) |
| "check my messages" | Reads iMessages via AppleScript |
| "what's on my calendar today" | Reads Calendar events |
| "reply saying I'll be there at 2" | Types and sends reply |
| "what is this on screen" | Claude sees your screen and describes it |
| "play music" / "volume up" | Media control (~80ms) |

---

## Configuration

All settings configurable via environment variables in `.env`:

```bash
# Required
GROQ_API_KEY=gsk_...

# Conversation mode (optional)
ANTHROPIC_API_KEY=sk-ant-...

# Customization (optional)
WHISPER_MODEL=distil-large-v3
GROQ_MODEL=llama-3.1-8b-instant
CONVERSATION_MODEL=claude-sonnet-4-20250514
TTS_VOICE=af_heart
TTS_SPEED=1.3
VOICE_AGENT_PORT=8528
```

Or set API keys from the Settings page inside the app.

---

## Troubleshooting

**Mic not working (RMS = 0.0000)**
- Run from **Terminal.app** (cmux/iTerm may lack mic permission)
- System Settings → Privacy & Security → Microphone → enable Terminal

**Hotkey not working**
- System Settings → Privacy & Security → Input Monitoring
- Add `/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app`

**Text not pasting**
- System Settings → Privacy & Security → Accessibility
- Text stays on clipboard — Cmd+V works as fallback

**Port 8528 in use**
- `pkill -f 'app.py'` then retry

**Conversation agent goes idle**
- Check terminal for error logs
- Ensure ANTHROPIC_API_KEY is set in .env

---

## Branches

| Branch | Purpose |
|---|---|
| `main` | Stable dictation mode |
| `conversational-control` | Conversation mode + desktop control |
| `browser-access` | Browser automation + web integration |

---

## License

MIT
