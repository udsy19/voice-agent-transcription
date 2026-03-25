"""Continuous Voice Activity Detection using Silero VAD.

Listens to the mic continuously, detects speech segments, and yields
complete utterances as numpy arrays ready for transcription.
"""

import asyncio
import time
import numpy as np
import sounddevice as sd
import threading
from collections import deque
from logger import get_logger

log = get_logger("vad")

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # silero VAD expects 512 samples at 16kHz
STOP_SECS = 0.8  # silence after speech before declaring end


class VADStream:
    def __init__(self, stop_secs: float = STOP_SECS):
        self._stop_secs = stop_secs
        self._running = False
        self._speech_active = False
        self._audio_buffer: list[np.ndarray] = []
        self._chunk_buffer = deque(maxlen=200)  # ~6s rolling buffer
        self._silence_start = 0.0
        self._model = None
        self._stream = None
        self._result_queue: asyncio.Queue = None
        self._loop = None

    @property
    def is_speech_active(self) -> bool:
        return self._speech_active

    def _load_model(self):
        """Load Silero VAD or fall back to energy-based VAD."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._get_speech_prob = lambda audio: float(
                model(torch.from_numpy(audio), SAMPLE_RATE).item()
            )
            log.info("Loaded Silero VAD (PyTorch)")
        except Exception as e:
            log.warning("Silero VAD failed (%s), using energy-based VAD", e)
            self._get_speech_prob = self._energy_vad

    def _energy_vad(self, audio: np.ndarray) -> float:
        """Simple energy-based VAD fallback."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        # Map RMS to 0-1 probability
        return min(1.0, rms / 0.02)

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            log.warning("Audio status: %s", status)
        chunk = indata[:, 0].copy().astype(np.float32)
        self._chunk_buffer.append(chunk)

    def _vad_loop(self):
        """Background thread: process chunks through VAD."""
        THRESHOLD = 0.5
        while self._running:
            if not self._chunk_buffer:
                time.sleep(0.01)
                continue

            chunk = self._chunk_buffer.popleft()
            prob = self._get_speech_prob(chunk)

            if prob >= THRESHOLD:
                if not self._speech_active:
                    self._speech_active = True
                    self._audio_buffer = []
                    log.debug("SPEECH_START")
                    if self._loop and self._result_queue:
                        asyncio.run_coroutine_threadsafe(
                            self._emit("speech_start"), self._loop
                        )
                self._audio_buffer.append(chunk)
                self._silence_start = 0.0
            else:
                if self._speech_active:
                    self._audio_buffer.append(chunk)  # include trailing silence
                    if self._silence_start == 0.0:
                        self._silence_start = time.time()
                    elif time.time() - self._silence_start >= self._stop_secs:
                        # Speech ended
                        self._speech_active = False
                        audio = np.concatenate(self._audio_buffer)
                        self._audio_buffer = []
                        self._silence_start = 0.0
                        log.debug("SPEECH_END (%.2fs)", len(audio) / SAMPLE_RATE)
                        if self._loop and self._result_queue:
                            asyncio.run_coroutine_threadsafe(
                                self._result_queue.put(audio), self._loop
                            )

    async def _emit(self, event: str):
        """Emit an event to the result queue."""
        pass  # Events are handled by checking is_speech_active

    async def start(self):
        """Start continuous listening. Yields complete utterances as numpy arrays."""
        self._load_model()
        self._running = True
        self._result_queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()

        # Get native mic rate
        try:
            info = sd.query_devices(kind="input")
            native_rate = int(info["default_samplerate"])
        except Exception:
            native_rate = 48000

        # Start mic stream
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        self._stream.start()
        log.info("VAD stream started (rate=%d)", SAMPLE_RATE)

        # Start VAD processing thread
        vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        vad_thread.start()

        # Yield complete utterances
        try:
            while self._running:
                try:
                    audio = await asyncio.wait_for(self._result_queue.get(), timeout=0.5)
                    yield audio
                except asyncio.TimeoutError:
                    continue
        finally:
            await self.stop()

    async def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        log.info("VAD stream stopped")
