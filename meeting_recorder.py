"""Meeting recorder — dual-stream audio capture with chunked transcription.

Records mic (You) and system audio via BlackHole (Them) simultaneously.
Chunks audio every 30s for live transcription, then does full summarization on stop.
"""

import os
import time
import json
import threading
import numpy as np
import sounddevice as sd
from datetime import datetime
from config import DATA_DIR
from logger import get_logger
from utils import detect_blackhole
import safe_json

log = get_logger("meeting")

MEETINGS_DIR = DATA_DIR / "meetings"
CHUNK_INTERVAL = 30  # seconds between transcription chunks
SAMPLE_RATE = 16000


def _resample(audio, orig_rate, target_rate):
    """Resample audio to target sample rate."""
    if orig_rate == target_rate:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_rate, target_rate)
    return resample_poly(audio, target_rate // g, orig_rate // g).astype(np.float32)


class MeetingRecorder:
    def __init__(self, transcriber, emit_fn):
        self._transcriber = transcriber
        self._emit = emit_fn
        self._recording = False
        self._meeting_id = None
        self._title = ""
        self._start_time = None
        self._calendar_event_id = None

        # Audio streams & buffers
        self._mic_stream = None
        self._sys_stream = None
        self._mic_frames = []
        self._sys_frames = []
        self._mic_lock = threading.Lock()
        self._sys_lock = threading.Lock()
        self._mic_rate = SAMPLE_RATE
        self._sys_rate = SAMPLE_RATE

        # Transcript
        self._chunks = []
        self._chunk_thread = None
        self._has_system_audio = False

    @property
    def is_recording(self):
        return self._recording

    def get_status(self):
        if not self._recording:
            return {"recording": False}
        elapsed = int(time.time() - self._start_time) if self._start_time else 0
        return {
            "recording": True,
            "meeting_id": self._meeting_id,
            "title": self._title,
            "start": datetime.fromtimestamp(self._start_time).isoformat() if self._start_time else "",
            "duration_seconds": elapsed,
            "has_system_audio": self._has_system_audio,
            "chunks": len(self._chunks),
        }

    def start(self, title="Meeting", calendar_event_id=None):
        """Start recording. Returns meeting_id."""
        if self._recording:
            return self._meeting_id

        MEETINGS_DIR.mkdir(parents=True, exist_ok=True)

        self._meeting_id = f"mtg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._title = title
        self._start_time = time.time()
        self._calendar_event_id = calendar_event_id
        self._chunks = []
        self._mic_frames = []
        self._sys_frames = []
        self._recording = True

        # Start mic stream
        self._start_mic()

        # Start system audio stream if BlackHole available
        bh_idx, bh_name = detect_blackhole()
        if bh_idx is not None:
            self._start_system(bh_idx, bh_name)
            self._has_system_audio = True
            log.info("Recording with system audio via %s", bh_name)
        else:
            self._has_system_audio = False
            log.info("BlackHole not detected — recording mic only")

        # Start chunk processor thread
        self._chunk_thread = threading.Thread(target=self._chunk_loop, daemon=True)
        self._chunk_thread.start()

        self._emit({"type": "meeting_started", "meeting_id": self._meeting_id, "title": title})
        log.info("Meeting recording started: %s (%s)", title, self._meeting_id)
        return self._meeting_id

    def stop(self):
        """Stop recording, finalize, return meeting data."""
        if not self._recording:
            return None

        self._recording = False
        end_time = time.time()

        # Stop streams
        if self._mic_stream:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        if self._sys_stream:
            try:
                self._sys_stream.stop()
                self._sys_stream.close()
            except Exception:
                pass
            self._sys_stream = None

        # Process any remaining audio in buffers
        self._process_chunk()

        # Build final meeting data
        duration = int(end_time - self._start_time)
        meeting = {
            "id": self._meeting_id,
            "title": self._title,
            "start": datetime.fromtimestamp(self._start_time).isoformat(),
            "end": datetime.fromtimestamp(end_time).isoformat(),
            "duration_minutes": round(duration / 60, 1),
            "calendar_event_id": self._calendar_event_id,
            "has_system_audio": self._has_system_audio,
            "chunks": self._chunks,
            "summary": None,
        }

        # Run summarization in background
        threading.Thread(target=self._summarize_and_save, args=(meeting,), daemon=True).start()

        self._emit({"type": "meeting_stopped", "meeting_id": self._meeting_id})
        log.info("Meeting stopped: %s (%.1f min, %d chunks)", self._title, duration / 60, len(self._chunks))
        return meeting

    def _start_mic(self):
        """Start the microphone input stream."""
        try:
            dev_info = sd.query_devices(kind="input")
            self._mic_rate = int(dev_info.get("default_samplerate", SAMPLE_RATE))
            self._mic_stream = sd.InputStream(
                samplerate=self._mic_rate, channels=1, dtype="float32",
                callback=self._mic_callback, blocksize=int(self._mic_rate * 0.1),
            )
            self._mic_stream.start()
        except Exception as e:
            log.error("Mic stream failed: %s", e)

    def _start_system(self, device_idx, device_name):
        """Start the system audio stream via BlackHole."""
        try:
            dev_info = sd.query_devices(device_idx)
            self._sys_rate = int(dev_info.get("default_samplerate", SAMPLE_RATE))
            self._sys_stream = sd.InputStream(
                device=device_idx, samplerate=self._sys_rate, channels=1,
                dtype="float32", callback=self._sys_callback,
                blocksize=int(self._sys_rate * 0.1),
            )
            self._sys_stream.start()
        except Exception as e:
            log.error("System audio stream failed: %s", e)
            self._has_system_audio = False

    def _mic_callback(self, indata, frames, time_info, status):
        with self._mic_lock:
            self._mic_frames.append(indata.copy())

    def _sys_callback(self, indata, frames, time_info, status):
        with self._sys_lock:
            self._sys_frames.append(indata.copy())

    def _chunk_loop(self):
        """Background thread: process audio chunks periodically."""
        while self._recording:
            time.sleep(CHUNK_INTERVAL)
            if not self._recording:
                break
            try:
                self._process_chunk()
            except Exception as e:
                log.error("Chunk processing error: %s", e)

    def _process_chunk(self):
        """Snapshot current buffers, transcribe, append to chunks."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Snapshot and clear mic buffer
        with self._mic_lock:
            mic_data = list(self._mic_frames)
            self._mic_frames.clear()

        # Snapshot and clear system buffer
        with self._sys_lock:
            sys_data = list(self._sys_frames)
            self._sys_frames.clear()

        # Transcribe mic audio (You)
        if mic_data:
            mic_audio = np.concatenate(mic_data).flatten()
            mic_audio = _resample(mic_audio, self._mic_rate, SAMPLE_RATE)
            mic_text = self._transcribe(mic_audio)
            if mic_text and mic_text.strip():
                chunk = {"timestamp": timestamp, "speaker": "You", "text": mic_text.strip()}
                self._chunks.append(chunk)
                self._emit({"type": "meeting_chunk", "meeting_id": self._meeting_id, **chunk})

        # Transcribe system audio (Them)
        if sys_data:
            sys_audio = np.concatenate(sys_data).flatten()
            sys_audio = _resample(sys_audio, self._sys_rate, SAMPLE_RATE)
            sys_text = self._transcribe(sys_audio)
            if sys_text and sys_text.strip():
                chunk = {"timestamp": timestamp, "speaker": "Them", "text": sys_text.strip()}
                self._chunks.append(chunk)
                self._emit({"type": "meeting_chunk", "meeting_id": self._meeting_id, **chunk})

    def _transcribe(self, audio):
        """Transcribe an audio numpy array using the app's transcriber."""
        if len(audio) < SAMPLE_RATE * 0.5:  # less than 0.5s
            return ""
        # Normalize
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.001:  # silence
            return ""
        if rms > 0 and rms < 0.1:
            audio = audio * (0.1 / rms)
        audio = np.clip(audio, -1.0, 1.0)
        try:
            return self._transcriber.transcribe(audio)
        except Exception as e:
            log.error("Meeting transcription error: %s", e)
            return ""

    def _summarize_and_save(self, meeting):
        """Run LLM summarization on the full transcript and save to disk."""
        # Known Whisper hallucination phrases to filter out
        _halluc_markers = [
            "thank you for watching", "thanks for watching", "subscribe to",
            "see you in the next", "subtitles by", "translated by",
        ]
        # Build full transcript — skip hallucinated chunks
        transcript_lines = []
        cleaned_chunks = []
        for c in meeting.get("chunks", []):
            text = (c.get("text") or "").strip()
            if not text or len(text) < 3:
                continue
            if any(m in text.lower() for m in _halluc_markers):
                continue
            speaker = c.get("speaker", "Unknown")
            timestamp = c.get("timestamp", "??:??:??")
            transcript_lines.append(f"[{timestamp}] {speaker}: {text}")
            cleaned_chunks.append(c)
        meeting["chunks"] = cleaned_chunks
        full_transcript = "\n".join(transcript_lines)

        # Summarize with LLM
        if full_transcript.strip():
            try:
                from llm import get_client
                client = get_client()
                resp = client.chat(
                    messages=[{
                        "role": "system",
                        "content": "You are a meeting summarizer. Given a meeting transcript, extract key information."
                    }, {
                        "role": "user",
                        "content": f"""Summarize this meeting transcript. Return JSON with:
- "key_points": array of 3-7 key discussion points
- "action_items": array of action items (who does what)
- "decisions": array of decisions made

Transcript:
{full_transcript[:8000]}

Respond ONLY with valid JSON."""
                    }],
                    model_tier="small",
                    temperature=0.3,
                    max_tokens=1024,
                )
                # Parse JSON from response
                text = resp.text.strip()
                # Handle markdown code blocks
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.startswith("json"):
                            text = text[4:]
                        text = text.strip()
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("Summary not a dict")
                # Validate + sanitize each field
                summary = {"key_points": [], "action_items": [], "decisions": []}
                for field in ("key_points", "action_items", "decisions"):
                    items = parsed.get(field, [])
                    if isinstance(items, list):
                        summary[field] = [str(x)[:500] for x in items[:20] if x and isinstance(x, (str, int, float))]
                meeting["summary"] = summary
            except Exception as e:
                log.error("Meeting summarization failed: %s", e)
                meeting["summary"] = {
                    "key_points": ["Summary unavailable — see full transcript"],
                    "action_items": [],
                    "decisions": [],
                    "_error": "summarization_failed",
                }
        else:
            meeting["summary"] = {
                "key_points": ["No speech detected in recording"],
                "action_items": [],
                "decisions": [],
            }

        # Save to disk
        path = MEETINGS_DIR / f"{meeting['id']}.json"
        safe_json.save(str(path), meeting)
        log.info("Meeting saved: %s", path)

        self._emit({"type": "meeting_complete", "meeting_id": meeting["id"], "title": meeting["title"]})


# ── File-based meeting storage ────────────────────────────────────────────

def list_meetings():
    """List all saved meetings, sorted by date descending."""
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    meetings = []
    for f in sorted(MEETINGS_DIR.glob("mtg_*.json"), reverse=True):
        try:
            data = safe_json.load(str(f), None)
            if data:
                meetings.append({
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "start": data.get("start", ""),
                    "end": data.get("end", ""),
                    "duration_minutes": data.get("duration_minutes", 0),
                    "has_system_audio": data.get("has_system_audio", False),
                    "chunk_count": len(data.get("chunks", [])),
                    "has_summary": data.get("summary") is not None,
                })
        except Exception:
            continue
    return meetings


def get_meeting(meeting_id):
    """Load a full meeting by ID."""
    path = MEETINGS_DIR / f"{meeting_id}.json"
    return safe_json.load(str(path), None)


def delete_meeting(meeting_id):
    """Delete a meeting file."""
    path = MEETINGS_DIR / f"{meeting_id}.json"
    try:
        path.unlink(missing_ok=True)
        return True
    except Exception:
        return False
