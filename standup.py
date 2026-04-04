"""Voice standup generator — ramble → structured update."""

from groq import Groq
from config import GROQ_API_KEY
from logger import get_logger

log = get_logger("standup")


def generate(ramble: str) -> str:
    """Transform rambling voice notes into a structured standup update."""
    if not GROQ_API_KEY:
        return ramble

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content":
                 "Transform the user's rambling notes into a clean standup update. "
                 "Format exactly as:\n\n"
                 "**Yesterday**\n- item\n\n"
                 "**Today**\n- item\n\n"
                 "**Blockers**\n- item (or 'None')\n\n"
                 "Be concise. Extract action items. Fix grammar. Keep the user's intent."},
                {"role": "user", "content": ramble},
            ],
            temperature=0.3,
            max_tokens=512,
            timeout=10,
        )
        result = response.choices[0].message.content.strip()
        log.info("Generated standup from %d words", len(ramble.split()))
        return result
    except Exception as e:
        log.error("Standup generation failed: %s", e)
        return ramble
