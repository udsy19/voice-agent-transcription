"""Streaming TTS using kokoro-onnx.

Speaks sentence-by-sentence for low latency (<300ms to first audio).
Interruptible: stops playback immediately when user starts speaking.
Falls back to macOS `say` command if kokoro is not available.
"""

import asyncio
import os
import re
import subprocess
import numpy as np
import sounddevice as sd
from pathlib import Path
from logger import get_logger
from config import DATA_DIR

log = get_logger("tts")

MODELS_DIR = DATA_DIR / "models"
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"
KOKORO_URLS = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+|(?<=\n)')
MARKDOWN_STRIP = re.compile(r'[*_`#~\[\]()]')  # remove markdown chars


def _download_models():
    """Download kokoro models if not present."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in KOKORO_URLS.items():
        path = MODELS_DIR / filename
        if path.exists():
            continue
        log.info("Downloading %s...", filename)
        try:
            import ssl
            import urllib.request
            # Try with certifi certs
            try:
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            with opener.open(url) as resp, open(path, 'wb') as f:
                f.write(resp.read())
            log.info("Downloaded %s (%.1f MB)", filename, path.stat().st_size / 1e6)
        except Exception as e:
            log.error("Failed to download %s: %s", filename, e)


class StreamingTTS:
    def __init__(self, voice: str = "af_heart"):
        self.voice = voice
        self.is_playing = False
        self.interrupted = False
        self._kokoro = None
        self._use_fallback = False
        self._load()

    def _load(self):
        try:
            _download_models()
            if KOKORO_MODEL.exists() and KOKORO_VOICES.exists():
                from kokoro_onnx import Kokoro
                self._kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
                log.info("Loaded Kokoro TTS (voice=%s)", self.voice)
            else:
                raise FileNotFoundError("Models not downloaded")
        except Exception as e:
            log.warning("Kokoro not available (%s), using macOS say", e)
            self._use_fallback = True

    async def speak(self, text: str):
        """Speak text sentence-by-sentence. Interruptible."""
        if not text:
            return

        # Strip markdown formatting
        text = MARKDOWN_STRIP.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return

        self.interrupted = False
        self.is_playing = True

        try:
            sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
            if not sentences:
                sentences = [text]

            for sentence in sentences:
                if self.interrupted:
                    break
                if self._use_fallback:
                    await self._speak_macos(sentence)
                else:
                    await self._speak_kokoro(sentence)
        finally:
            self.is_playing = False

    async def _speak_kokoro(self, text: str):
        """Generate and play audio for one sentence via kokoro."""
        try:
            # Emotional speed: adjust based on content
            speed = self._detect_speed(text)
            samples, sample_rate = self._kokoro.create(
                text, voice=self.voice, speed=speed
            )
            if self.interrupted:
                return

            # Play audio
            sd.play(samples, sample_rate)

            # Wait for playback to finish (check for interruption)
            duration = len(samples) / sample_rate
            elapsed = 0
            while elapsed < duration and not self.interrupted:
                await asyncio.sleep(0.05)
                elapsed += 0.05

            if self.interrupted:
                sd.stop()
        except Exception as e:
            log.error("Kokoro speak failed: %s", e)
            await self._speak_macos(text)

    async def _speak_macos(self, text: str):
        """Fallback: use macOS say command."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "say", "-r", "250", text,  # faster macOS TTS
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            while proc.returncode is None and not self.interrupted:
                await asyncio.sleep(0.05)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
            if self.interrupted and proc.returncode is None:
                proc.kill()
        except Exception as e:
            log.error("macOS say failed: %s", e)

    def _detect_speed(self, text: str) -> float:
        """Adjust TTS speed based on content emotion."""
        lower = text.lower()
        # Excited/positive
        if any(w in lower for w in ["!", "great", "awesome", "perfect", "done", "got it", "sure"]):
            return 1.4
        # Serious/important
        if any(w in lower for w in ["important", "warning", "careful", "error", "failed", "sorry"]):
            return 1.1
        # Informational/reading back data
        if any(w in lower for w in ["you have", "there are", "showing", "from", "subject"]):
            return 1.2
        # Default
        return 1.3

    def interrupt(self):
        """Stop playback immediately."""
        self.interrupted = True
        self.is_playing = False
        try:
            sd.stop()
        except Exception:
            pass
