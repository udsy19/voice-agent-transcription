"""Shared Google API service builder — cached to avoid rebuilding on every call."""

import time
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from logger import get_logger

log = get_logger("gauth")

_service_cache = {}  # {(api, version, token): (service, ts)}
CACHE_TTL = 1800  # 30 min


def get_service(token_data: dict, api: str, version: str):
    """Build a Google API service, cached per token+api for 30 min."""
    token = token_data.get("access_token", "")
    cache_key = (api, version, token[:20])

    cached = _service_cache.get(cache_key)
    if cached and (time.time() - cached[1]) < CACHE_TTL:
        return cached[0]

    if "credentials" in token_data:
        creds = token_data["credentials"]
    else:
        creds = Credentials(
            token=token,
            refresh_token=token_data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
        )

    service = build(api, version, credentials=creds, cache_discovery=False)
    _service_cache[cache_key] = (service, time.time())
    return service


def api_error(e) -> dict:
    """Convert Google API errors to user-friendly messages."""
    err = str(e)
    if "accessNotConfigured" in err or "has not been used" in err:
        api_name = "Google Calendar" if "calendar" in err.lower() else "Gmail"
        return {"ok": False, "error": f"{api_name} API not enabled. Enable at console.cloud.google.com."}
    return {"ok": False, "error": err}
