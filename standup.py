"""Voice standup generator — ramble → structured update."""

from logger import get_logger

log = get_logger("standup")


def generate(ramble: str) -> str:
    """Transform rambling voice notes into a structured standup update."""
    try:
        from llm import get_client
        client = get_client()
        response = client.chat(
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
            model_tier="small",
            temperature=0.3,
            max_tokens=512,
            timeout=10,
        )
        result = response.text.strip()
        if result:
            log.info("Generated standup from %d words", len(ramble.split()))
            return result
        return ramble
    except Exception as e:
        log.error("Standup generation failed: %s", e)
        return ramble
