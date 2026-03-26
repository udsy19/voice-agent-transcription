import numpy as np
import sounddevice as sd
import threading
from config import SAMPLE_RATE, CHANNELS, DTYPE, SILENCE_THRESHOLD, MIN_AUDIO_DURATION
from logger import get_logger

log = get_logger("recorder")

MAX_RECORDING_SECONDS = 300


# ── Voice Isolation (DeepFilterNet) ──────────────────────────────────────────

_df_model = None
_df_state = None
_df_available = None  # None = not checked, True/False = checked


def _init_voice_isolation():
    """Lazy-load DeepFilterNet for voice isolation."""
    global _df_model, _df_state, _df_available
    if _df_available is not None:
        return _df_available
    try:
        from df.enhance import init_df
        _df_model, _df_state, _ = init_df()
        _df_available = True
        log.info("Voice isolation loaded (DeepFilterNet3, %dHz)", _df_state.sr())
        return True
    except Exception as e:
        _df_available = False
        log.info("Voice isolation not available: %s", e)
        return False


def _apply_voice_isolation(audio: np.ndarray) -> np.ndarray:
    """Apply DeepFilterNet noise suppression to 48kHz audio."""
    if not _df_available or _df_model is None:
        return audio
    try:
        import torch
        from df.enhance import enhance
        tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]
        enhanced = enhance(_df_model, _df_state, tensor)
        if isinstance(enhanced, torch.Tensor):
            return enhanced.squeeze().numpy().astype(np.float32)
        return enhanced.astype(np.float32)
    except Exception as e:
        log.debug("Voice isolation failed: %s", e)
        return audio


def _get_native_rate() -> int:
    try:
        info = sd.query_devices(kind="input")
        return int(info["default_samplerate"])
    except Exception as e:
        log.warning("Could not detect mic rate: %s, defaulting to 48000", e)
        return 48000


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if orig_rate == target_rate:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_rate, target_rate)
        up, down = target_rate // g, orig_rate // g
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception as e:
        log.error("Resample failed: %s, using linear interpolation", e)
        # Fallback: simple linear interpolation
        target_len = int(len(audio) * target_rate / orig_rate)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class Recorder:
    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._recording = False
        self._lock = threading.Lock()
        self._whisper_mode = False
        self._voice_isolation = False
        self._total_samples = 0
        try:
            self._native_rate = _get_native_rate()
        except Exception:
            self._native_rate = 48000
        log.info("Mic native rate: %d Hz, resample to %d Hz", self._native_rate, SAMPLE_RATE)

    @property
    def is_recording(self):
        return self._recording

    @property
    def whisper_mode(self):
        return self._whisper_mode

    def toggle_whisper_mode(self):
        self._whisper_mode = not self._whisper_mode
        log.info("Whisper mode %s", "ON" if self._whisper_mode else "OFF")
        return self._whisper_mode

    @property
    def voice_isolation(self):
        return self._voice_isolation

    def toggle_voice_isolation(self):
        if not _init_voice_isolation():
            log.warning("Voice isolation not available")
            return False
        self._voice_isolation = not self._voice_isolation
        log.info("Voice isolation %s", "ON" if self._voice_isolation else "OFF")
        return self._voice_isolation

    def set_voice_isolation(self, enabled: bool):
        if enabled and not _init_voice_isolation():
            log.warning("Voice isolation not available")
            return False
        self._voice_isolation = enabled
        log.info("Voice isolation %s", "ON" if enabled else "OFF")
        return enabled

    def start(self):
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._total_samples = 0
            self._recording = True
            try:
                self._stream = sd.InputStream(
                    samplerate=self._native_rate,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    callback=self._audio_callback,
                )
                self._stream.start()
                log.info("Recording started%s", " (whisper mode)" if self._whisper_mode else "")
            except Exception as e:
                log.error("Failed to start recording: %s", e)
                self._recording = False
                self._stream = None

    def stop(self) -> np.ndarray | None:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    log.warning("Error stopping stream: %s", e)
                self._stream = None

            # Grab frames under lock to prevent race with callback
            frames = self._frames.copy()
            self._frames = []

        if not frames:
            log.debug("No frames captured")
            return None

        try:
            audio = np.concatenate(frames, axis=0).flatten()
        except Exception as e:
            log.error("Failed to concatenate frames: %s", e)
            return None

        # Check for NaN/Inf
        if not np.isfinite(audio).all():
            log.warning("Audio contains NaN/Inf, cleaning")
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        duration = len(audio) / self._native_rate
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Handle NaN RMS
        if np.isnan(rms):
            log.warning("RMS is NaN")
            return None

        threshold = SILENCE_THRESHOLD * 0.1 if self._whisper_mode else SILENCE_THRESHOLD
        log.info("Captured %.2fs, RMS=%.4f, threshold=%.4f", duration, rms, threshold)

        if duration < MIN_AUDIO_DURATION:
            log.info("Too short, ignoring")
            return None
        if rms < threshold:
            log.info("Too quiet (silence), ignoring")
            return None

        if self._whisper_mode and rms < 0.05:
            gain = min(0.05 / max(rms, 1e-6), 20.0)
            audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
            log.info("Whisper mode: amplified %.1fx", gain)

        # Voice isolation: remove background noise at native rate (before resampling)
        if self._voice_isolation:
            import time as _t
            t0 = _t.time()
            audio = _apply_voice_isolation(audio)
            log.info("Voice isolation: %.0fms", (_t.time() - t0) * 1000)

        audio = _resample(audio, self._native_rate, SAMPLE_RATE)
        log.info("Resampled to %d Hz, %d samples", SAMPLE_RATE, len(audio))
        return audio

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning("Stream status: %s", status)
        if not self._recording:
            return
        if self._total_samples / self._native_rate > MAX_RECORDING_SECONDS:
            return
        with self._lock:
            self._frames.append(indata.copy())
            self._total_samples += indata.shape[0]
