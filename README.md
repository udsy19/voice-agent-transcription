# Voice Agent

Local voice dictation that works system-wide. Hold a key, speak, release — cleaned text appears wherever your cursor is.

Runs entirely on your machine. No audio leaves your device (except the optional Groq API call for text cleanup).

![Hold to dictate](https://img.shields.io/badge/Hold_⌥R-Dictate-1C0E04?style=for-the-badge)
![Local Whisper](https://img.shields.io/badge/Whisper-Local_ASR-CC8A24?style=for-the-badge)
![Groq Cleanup](https://img.shields.io/badge/Groq-LLM_Cleanup-877263?style=for-the-badge)

---

## How it works

1. **Hold Right Option (⌥)** — recording starts, floating pill shows waveform
2. **Speak** — Whisper transcribes locally in real-time
3. **Release** — text gets cleaned by Groq and pasted at your cursor
4. Text also stays on clipboard for Cmd+V

Works in any app — Slack, Notion, VS Code, browser, terminal, anywhere.

---

## Features

**Core**
- Hold-to-record with global hotkey (Right Option)
- Local transcription via Whisper (distil-large-v3)
- AI cleanup: removes fillers (um, uh, like), fixes grammar, adds punctuation
- Pastes directly at cursor position, also copies to clipboard

**Smart**
- Backtracking — "Tuesday... actually Wednesday" → "Wednesday"
- List formatting — "first X second Y third Z" → numbered list
- Smart undo — say "undo that" within 10s to revert
- Command mode — "Hey Flow, make this more professional" transforms selected text
- Streaming preview — partial text appears in pill as you speak

**Personalization**
- Auto-learning dictionary — detects names, emails, acronyms and remembers them
- Manual dictionary with custom corrections
- Voice snippets — "insert email signature" → pastes full text block
- Voice macros — "email mode" chains actions (set tone + insert template)
- Per-app tone — formal in Docs, casual in Slack, code syntax in VS Code

**Desktop App**
- Electron app with floating pill (always visible, expands on recording)
- Sidebar UI: Home, Dictionary, Snippets, Style, Macros, Scratchpad, Settings
- Domain modes: Tech, Medical, Legal, Finance (specialized vocabulary)
- Style per context: Personal messages, Work messages, Email, Other
- Tray icon, runs in background

---

## Quick Start

### Prerequisites

- **macOS** (Apple Silicon or Intel)
- **Python 3.11+**
- **Node.js 18+**
- **Groq API key** — get one free at [console.groq.com](https://console.groq.com)

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
# From Terminal.app (needed for mic access)
bash start.sh
```

This starts the Python backend + Electron UI. The floating pill appears at the top of your screen.

**First time:** macOS will ask for Microphone and Accessibility permissions. Grant both.

You can also set your Groq API key from Settings inside the app (no need to edit `.env`).

### Alternative: Web-only mode

```bash
python3 app.py
```

Opens a browser dashboard at `http://localhost:8528`. Same features, no Electron needed.

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Electron (UI)                          │
│  ├── Main window (settings, history)    │
│  ├── Floating pill (waveform, status)   │
│  └── Tray icon                          │
│           │                             │
│     WebSocket + REST                    │
│           │                             │
│  Python Backend (localhost:8528)         │
│  ├── pynput (global hotkey listener)    │
│  ├── sounddevice (mic capture)          │
│  ├── faster-whisper (local ASR)         │
│  ├── Groq API (LLM text cleanup)       │
│  └── Quartz CGEvent (text injection)    │
└─────────────────────────────────────────┘
```

**Privacy:** Audio is transcribed locally by Whisper. Only the transcribed text is sent to Groq for cleanup (grammar, punctuation, filler removal). No audio ever leaves your machine.

---

## File Structure

```
├── app.py              # FastAPI server + voice engine
├── recorder.py         # Mic capture, silence detection, resampling
├── transcriber.py      # Whisper wrapper (streaming + batch)
├── cleaner.py          # Groq LLM cleanup + auto-term extraction
├── injector.py         # Text injection (paste + character typing)
├── dictionary.py       # Personal dictionary (terms + corrections)
├── snippets.py         # Voice-triggered text snippets
├── macros.py           # Chainable voice workflows
├── domains.py          # Domain-specific vocabulary modes
├── styles.py           # Tone profiles and per-app overrides
├── config.py           # All configuration
├── logger.py           # Rotating file logger
├── start.sh            # Launcher script
├── electron/
│   ├── main.js         # Electron main process
│   ├── preload.js      # Context bridge
│   └── ui/
│       ├── app.html    # Settings window
│       └── pill.html   # Floating pill overlay
└── .env                # GROQ_API_KEY (not committed)
```

---

## Voice Commands

| Say this | What happens |
|---|---|
| *(hold ⌥R and speak)* | Dictates and pastes text |
| "undo that" / "go back" | Reverts the last paste (within 10s) |
| "Hey Flow, make this more professional" | Transforms selected text |
| "email mode" | Sets formal tone + inserts greeting |
| "code mode" | Sets code tone + activates tech domain |
| "standup notes" | Inserts standup template |
| "meeting notes" | Inserts meeting notes template |
| "insert *(snippet trigger)*" | Pastes the matching snippet |
| "scratch that" | Deletes the preceding sentence |
| "new paragraph" | Inserts a line break |
| "period" / "comma" / "question mark" | Inserts punctuation |

---

## Configuration

### Groq API Key

Set via the Settings page in the app, or manually:

```bash
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### Whisper Model

Edit `config.py`:

```python
WHISPER_MODEL = "distil-large-v3"  # best accuracy
# or "distil-small.en"            # faster, slightly less accurate
```

### Silence Threshold

If recordings are being rejected as "too quiet":

```python
SILENCE_THRESHOLD = 0.005  # lower = more sensitive (default)
```

---

## Troubleshooting

**Mic not working (RMS = 0.0000)**
- Run from **Terminal.app**, not other terminals (cmux, iTerm may lack mic permission)
- Check System Settings → Privacy & Security → Microphone

**Text not pasting**
- Check System Settings → Privacy & Security → Accessibility
- Text is always on clipboard — you can Cmd+V manually

**Port 8528 in use**
- `pkill -f 'app.py'` then retry

**Slow transcription**
- Try `distil-small.en` model in config.py (2-3x faster, slightly less accurate)
- Or use Groq's cloud Whisper API for near-instant transcription

---

## License

MIT
