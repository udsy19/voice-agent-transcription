import os
import numpy as np
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
from logger import get_logger

log = get_logger("transcriber")


def _get_optimal_threads():
    try:
        cores = os.cpu_count() or 4
        return max(4, int(cores * 0.75))
    except Exception:
        return 4


class Transcriber:
    def __init__(self):
        threads = _get_optimal_threads()
        log.info("Loading model '%s' (%d threads)...", WHISPER_MODEL, threads)
        self._model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=threads,
            num_workers=2,
        )
        log.info("Model loaded")

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   initial_prompt: str | None = None) -> str:
        """Transcribe audio, return full text."""
        if audio is None or len(audio) == 0:
            log.warning("Empty audio, skipping")
            return ""
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        segments, info = self._model.transcribe(
            audio,
            beam_size=1, best_of=1,
            language=language or "en",
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=150),
            without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        detected = info.language if hasattr(info, "language") else "?"
        log.info("(%s) '%s'", detected, text)
        return text.strip()

    def transcribe_streaming(self, audio: np.ndarray, on_segment=None,
                             language: str | None = None,
                             initial_prompt: str | None = None) -> str:
        """Transcribe audio, calling on_segment(partial_text) as each segment completes."""
        if audio is None or len(audio) == 0:
            return ""
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        segments, info = self._model.transcribe(
            audio,
            beam_size=1, best_of=1,
            language=language or "en",
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=150),
            without_timestamps=True,
        )

        parts = []
        for seg in segments:
            parts.append(seg.text.strip())
            if on_segment:
                on_segment(" ".join(parts))

        text = " ".join(parts)
        detected = info.language if hasattr(info, "language") else "?"
        log.info("(%s) '%s'", detected, text)
        return text.strip()
