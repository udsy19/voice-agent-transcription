"""Screen-aware commands — screenshot + Claude Vision API."""

import os
import base64
import subprocess
import tempfile
from logger import get_logger

log = get_logger("vision")


def take_screenshot() -> bytes | None:
    """Capture the active screen to PNG bytes."""
    try:
        tmp = os.path.join(tempfile.gettempdir(), "muse_screenshot.png")
        subprocess.run(["screencapture", "-x", "-C", tmp], capture_output=True, timeout=5)
        if os.path.exists(tmp):
            with open(tmp, "rb") as f:
                data = f.read()
            os.remove(tmp)
            return data
    except Exception as e:
        log.error("Screenshot failed: %s", e)
    return None


def analyze_screen(instruction: str) -> dict:
    """Screenshot the screen and analyze with Claude Vision."""
    from config import GROQ_API_KEY

    # Get Anthropic key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        from utils import keychain_get
        anthropic_key = keychain_get("Muse", "anthropic_api_key")

    img_bytes = take_screenshot()
    if not img_bytes:
        return {"ok": False, "error": "Screenshot failed"}

    if not anthropic_key:
        # Fallback: describe what we can see without vision
        return {"ok": False, "error": "No Anthropic API key. Add ANTHROPIC_API_KEY to .env for screen analysis."}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        img_b64 = base64.b64encode(img_bytes).decode()

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": f"Look at this screenshot and {instruction}. Be concise — 2-3 sentences max. If asked to summarize, summarize the main content. If asked to reply, draft a short reply."},
                ],
            }],
        )
        text = response.content[0].text
        log.info("Vision: %s → %s", instruction[:40], text[:60])
        return {"ok": True, "analysis": text}

    except ImportError:
        return {"ok": False, "error": "anthropic package not installed. Run: pip install anthropic"}
    except Exception as e:
        log.error("Vision failed: %s", e)
        return {"ok": False, "error": str(e)}
