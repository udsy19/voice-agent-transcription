"""Model manager — download, cache, and verify LLM + TTS models.

Models are stored in ~/Library/Application Support/Muse/models/
Downloaded on first use, cached permanently.
"""

import os
import threading
from config import DATA_DIR
from logger import get_logger

log = get_logger("models")

MODELS_DIR = DATA_DIR / "models"
LLM_DIR = MODELS_DIR / "llm"
TTS_DIR = MODELS_DIR  # Kokoro models already use this


# ── Model registry ────────────────────────────────────────────────────────

MODELS = {
    "llm": {
        "name": "Mistral 7B Instruct (4-bit)",
        "hf_repo": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "size_mb": 4200,
        "required": False,
        "description": "Local AI model for offline assistant and text cleanup",
    },
    "llm_small": {
        "name": "Qwen 3B Instruct (4-bit)",
        "hf_repo": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "size_mb": 1800,
        "required": False,
        "description": "Smaller local model — faster but less capable",
    },
    "tts": {
        "name": "Kokoro TTS v1.0",
        "files": ["kokoro-v1.0.onnx", "voices-v1.0.bin"],
        "size_mb": 337,
        "required": False,
        "description": "Local text-to-speech with 12 voices",
    },
    "whisper": {
        "name": "Whisper (faster-whisper)",
        "hf_repo": "Systran/faster-distil-whisper-large-v3",
        "size_mb": 1500,
        "required": False,
        "description": "Local speech-to-text transcription",
    },
}


def is_model_downloaded(model_key: str) -> bool:
    """Check if a model is already downloaded."""
    info = MODELS.get(model_key)
    if not info:
        return False

    if model_key in ("llm", "llm_small", "whisper"):
        # HuggingFace models — check if the repo is cached
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            for repo in cache.repos:
                if repo.repo_id == info["hf_repo"]:
                    return True
        except Exception:
            pass
        return False

    elif model_key == "tts":
        return all(os.path.exists(TTS_DIR / f) for f in info["files"])

    return False


def get_models_status() -> dict:
    """Get status of all models."""
    result = {}
    for key, info in MODELS.items():
        result[key] = {
            "name": info["name"],
            "description": info["description"],
            "size_mb": info["size_mb"],
            "downloaded": is_model_downloaded(key),
            "required": info["required"],
        }
    return result


_download_progress: dict = {}  # key -> {"progress": float, "total_mb": int, "status": str}
_download_lock = threading.Lock()


def download_model(model_key: str, progress_callback=None) -> bool:
    """Download a model. Blocks until complete. Returns True on success."""
    info = MODELS.get(model_key)
    if not info:
        log.error("Unknown model: %s", model_key)
        return False

    if is_model_downloaded(model_key):
        log.info("Model %s already downloaded", model_key)
        return True

    with _download_lock:
        _download_progress[model_key] = {"progress": 0, "total_mb": info["size_mb"], "status": "downloading"}

    try:
        if model_key in ("llm", "llm_small", "whisper"):
            return _download_hf_model(model_key, info, progress_callback)
        elif model_key == "tts":
            log.info("TTS models download automatically on first use")
            return True
    except Exception as e:
        log.error("Download failed for %s: %s", model_key, e)
        _download_progress[model_key] = {"progress": 0, "total_mb": info["size_mb"], "status": f"error: {e}"}
        return False


def _get_hf_cache_size(repo_id: str) -> int:
    """Get current downloaded size of a HF repo in bytes."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                return repo.size_on_disk
    except Exception:
        pass
    # Fallback: scan blob dir
    try:
        import pathlib
        cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
        if repo_dir.exists():
            return sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
    except Exception:
        pass
    return 0


def _download_hf_model(model_key: str, info: dict, progress_callback=None) -> bool:
    """Download a HuggingFace model with real progress tracking."""
    repo = info["hf_repo"]
    total_bytes = info["size_mb"] * 1024 * 1024
    log.info("Downloading %s from %s ...", info["name"], repo)

    # Start a progress monitor thread that checks cache size
    _progress_stop = threading.Event()

    def monitor_progress():
        while not _progress_stop.is_set():
            try:
                current = _get_hf_cache_size(repo)
                pct = min(current / total_bytes, 0.99) if total_bytes > 0 else 0
                _download_progress[model_key] = {
                    "progress": pct, "total_mb": info["size_mb"],
                    "downloaded_mb": round(current / (1024 * 1024)),
                    "status": "downloading",
                }
                if progress_callback:
                    progress_callback(pct)
            except Exception:
                pass
            _progress_stop.wait(2)  # check every 2 seconds

    monitor = threading.Thread(target=monitor_progress, daemon=True)
    monitor.start()

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo,
            local_dir=None,  # use default cache
            resume_download=True,
        )
        _progress_stop.set()
        _download_progress[model_key] = {"progress": 1.0, "total_mb": info["size_mb"], "status": "complete"}
        log.info("Downloaded: %s", info["name"])
        if progress_callback:
            progress_callback(1.0)
        return True
    except Exception as e:
        _progress_stop.set()
        log.error("HuggingFace download failed: %s", e)
        return False


def get_download_progress(model_key: str) -> dict:
    """Get download progress for a model."""
    return _download_progress.get(model_key, {"progress": 0, "total_mb": 0, "status": "idle"})


def download_model_async(model_key: str, emit_fn=None):
    """Download a model in background thread. Emits real-time progress via WebSocket."""
    def run():
        last_pct = -1

        def on_progress(pct):
            nonlocal last_pct
            # Only emit if progress changed by at least 1%
            rounded = round(pct * 100)
            if emit_fn and rounded != last_pct:
                last_pct = rounded
                info = MODELS.get(model_key, {})
                prog = _download_progress.get(model_key, {})
                emit_fn({"type": "model_download", "model": model_key,
                         "progress": pct, "total_mb": info.get("size_mb", 0),
                         "downloaded_mb": prog.get("downloaded_mb", 0)})

        success = download_model(model_key, progress_callback=on_progress)
        if emit_fn:
            emit_fn({"type": "model_download", "model": model_key,
                     "progress": 1.0 if success else -1,
                     "status": "complete" if success else "failed"})

    threading.Thread(target=run, daemon=True).start()
