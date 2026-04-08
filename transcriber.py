"""Multi-backend transcriber.

Supports:
  - "faster-whisper"  — CPU-based, distil-large-v3 (default fallback)
  - "mlx"             — Apple Silicon GPU via MLX, large-v3-turbo
  - "groq"            — Cloud API, whisper-large-v3 (fastest, needs internet)
  - "parakeet"        — Apple Silicon GPU via MLX, 0.6B params (fastest local)

Backend is selectable at runtime via set_backend().
"""

import os
import io
import time
import tempfile
import atexit
import numpy as np
import soundfile as sf
from logger import get_logger

log = get_logger("transcriber")

BACKENDS = ["parakeet", "mlx", "groq", "faster-whisper"]

# Track temp files for cleanup
_temp_files: list[str] = []

def _cleanup_temp_files():
    for path in _temp_files:
        try:
            os.remove(path)
        except Exception:
            pass
    # Also clean any stale temp files from previous crashes
    import glob
    for f in glob.glob(os.path.join(tempfile.gettempdir(), "muse_*.wav")):
        try:
            os.remove(f)
        except Exception:
            pass

atexit.register(_cleanup_temp_files)
# Clean stale files from previous runs on import
_cleanup_temp_files()


def _get_optimal_threads():
    try:
        return max(4, int((os.cpu_count() or 4) * 0.75))
    except Exception:
        return 4


def _audio_to_wav(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Write audio array to a temporary WAV file. Returns path."""
    fd, path = tempfile.mkstemp(suffix='.wav', prefix='muse_')
    os.close(fd)
    sf.write(path, audio, sample_rate)
    _temp_files.append(path)
    return path


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> io.BytesIO:
    """Write audio to in-memory WAV buffer."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format='WAV')
    buf.seek(0)
    return buf


def _clean_audio(audio: np.ndarray) -> np.ndarray:
    """Sanitize audio: fix NaN/Inf, normalize if clipping."""
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    # Normalize if audio is clipping (peak > 0.95)
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio * (0.9 / peak)
    return audio


class Transcriber:
    def __init__(self, backend: str = "parakeet"):
        self._backend_name = backend
        self._model = None
        self._groq_client = None
        self._consecutive_failures = 0
        self._load_backend(backend)

    def _load_backend(self, backend: str):
        self._backend_name = backend
        self._model = None
        t0 = time.time()

        if backend == "parakeet":
            try:
                import parakeet_mlx
                self._model = parakeet_mlx.from_pretrained('mlx-community/parakeet-tdt-0.6b-v2')
                self._backend_name = "parakeet"
                log.info("Loaded Parakeet MLX (%.1fs)", time.time() - t0)
                return
            except Exception as e:
                log.warning("Parakeet failed to load: %s, falling back to mlx", e)
                backend = "mlx"

        if backend == "mlx":
            try:
                import mlx_whisper
                self._model = mlx_whisper
                self._backend_name = "mlx"
                log.info("Loaded MLX Whisper (%.1fs)", time.time() - t0)
                return
            except ImportError:
                log.warning("mlx-whisper not installed, falling back to faster-whisper")
                backend = "faster-whisper"

        if backend == "groq":
            try:
                from groq import Groq
                from config import GROQ_API_KEY
                if GROQ_API_KEY:
                    self._groq_client = Groq(api_key=GROQ_API_KEY)
                    self._backend_name = "groq"
                    log.info("Loaded Groq Whisper API")
                    return
                else:
                    log.warning("No Groq API key for transcription, falling back")
                    backend = "faster-whisper"
            except ImportError:
                backend = "faster-whisper"

        # Default: faster-whisper (CPU)
        from faster_whisper import WhisperModel
        from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
        threads = _get_optimal_threads()
        log.info("Loading faster-whisper '%s' (%d threads)...", WHISPER_MODEL, threads)
        self._model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=threads, num_workers=2,
        )
        self._backend_name = "faster-whisper"
        log.info("Loaded faster-whisper (%.1fs)", time.time() - t0)

    def set_backend(self, backend: str):
        """Switch transcription backend at runtime."""
        if backend == self._backend_name:
            return
        log.info("Switching backend: %s -> %s", self._backend_name, backend)
        self._load_backend(backend)

    @property
    def backend(self) -> str:
        return self._backend_name

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   initial_prompt: str | None = None) -> str:
        if audio is None or len(audio) == 0:
            return ""
        audio = _clean_audio(audio)
        t0 = time.time()

        if self._backend_name == "parakeet":
            text = self._transcribe_parakeet(audio)
        elif self._backend_name == "mlx":
            text = self._transcribe_mlx(audio, language, initial_prompt)
        elif self._backend_name == "groq":
            text = self._transcribe_groq(audio, language, initial_prompt)
        else:
            text = self._transcribe_faster_whisper(audio, language, initial_prompt)

        log.info("[%s] (%.2fs) '%s'", self._backend_name, time.time() - t0, text[:100])
        return text

    def transcribe_streaming(self, audio: np.ndarray, on_segment=None,
                             language: str | None = None,
                             initial_prompt: str | None = None) -> str:
        """Streaming transcription — calls on_segment(partial_text) as available."""
        if self._backend_name == "faster-whisper":
            return self._transcribe_faster_whisper_streaming(audio, on_segment, language, initial_prompt)
        # Other backends don't support true streaming, just return full result
        text = self.transcribe(audio, language, initial_prompt)
        if on_segment and text:
            on_segment(text)
        return text

    # ── Backend implementations ──────────────────────────────────────────────

    def _transcribe_parakeet(self, audio: np.ndarray) -> str:
        try:
            wav_path = _audio_to_wav(audio)
            result = self._model.transcribe(wav_path)
            if hasattr(result, 'text'):
                return result.text.strip()
            return str(result).strip()
        except Exception as e:
            log.error("Parakeet failed: %s, falling back", e)
            return self._transcribe_fallback(audio)

    def _transcribe_mlx(self, audio: np.ndarray, language=None, prompt=None) -> str:
        try:
            wav_path = _audio_to_wav(audio)
            result = self._model.transcribe(
                wav_path,
                path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                language=language,
                initial_prompt=prompt,
            )
            return result.get("text", "").strip()
        except Exception as e:
            log.error("MLX failed: %s, falling back", e)
            return self._transcribe_fallback(audio)

    def _transcribe_groq(self, audio: np.ndarray, language=None, prompt=None) -> str:
        try:
            buf = _audio_to_wav_bytes(audio)
            kwargs = {
                "model": "whisper-large-v3-turbo",
                "file": ("audio.wav", buf, "audio/wav"),
                "language": language or "en",
                "response_format": "verbose_json",
                "temperature": 0.0,
            }
            if prompt:
                kwargs["prompt"] = prompt
            response = self._groq_client.audio.transcriptions.create(**kwargs)
            self._consecutive_failures = 0
            if hasattr(response, 'text'):
                return response.text.strip()
            return str(response).strip()
        except Exception as e:
            self._consecutive_failures = getattr(self, '_consecutive_failures', 0) + 1
            log.error("Groq failed (%d consecutive): %s", self._consecutive_failures, e)
            if self._consecutive_failures >= 3:
                log.warning("Groq failed 3x — auto-switching to local faster-whisper")
                self._load_backend("faster-whisper")
                self._consecutive_failures = 0
            return self._transcribe_fallback(audio)

    def _transcribe_faster_whisper(self, audio, language=None, prompt=None) -> str:
        segments, info = self._model.transcribe(
            audio, beam_size=3, best_of=1,
            language=language,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
                threshold=0.35,
            ),
            without_timestamps=True,
        )
        if info and hasattr(info, 'language'):
            log.info("Detected language: %s (%.0f%%)", info.language, (info.language_probability or 0) * 100)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _transcribe_faster_whisper_streaming(self, audio, on_segment, language=None, prompt=None) -> str:
        audio = _clean_audio(audio)
        segments, info = self._model.transcribe(
            audio, beam_size=3, best_of=1,
            language=language,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
                threshold=0.35,
            ),
            without_timestamps=True,
        )
        if info and hasattr(info, 'language'):
            log.info("Detected language: %s (%.0f%%)", info.language, (info.language_probability or 0) * 100)
        parts = []
        for seg in segments:
            parts.append(seg.text.strip())
            if on_segment:
                on_segment(" ".join(parts))
        return " ".join(parts).strip()

    def _transcribe_fallback(self, audio) -> str:
        """Last resort: load faster-whisper on the fly."""
        try:
            from faster_whisper import WhisperModel
            from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
            model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                                 compute_type=WHISPER_COMPUTE_TYPE,
                                 cpu_threads=_get_optimal_threads())
            segments, _ = model.transcribe(audio, beam_size=3, best_of=1,
                                           vad_filter=True, without_timestamps=True)
            return " ".join(seg.text.strip() for seg in segments).strip()
        except ImportError:
            log.error("faster-whisper not installed — no fallback available")
            return "[transcription failed — no fallback engine]"
        except Exception as e:
            log.error("All transcription backends failed: %s", e)
            return "[transcription failed]"
