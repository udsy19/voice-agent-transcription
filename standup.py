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
        if not result:
            return ramble
        # Validate structure: must have Yesterday + Today sections
        rl = result.lower()
        if "yesterday" not in rl or "today" not in rl:
            log.warning("Standup missing required sections — using raw ramble")
            return ramble
        # Cross-check: output should share vocab with input (not a rewrite)
        ramble_words = set(w.lower().strip(".,!?") for w in ramble.split() if len(w) > 3)
        result_words = set(w.lower().strip(".,!?") for w in result.split() if len(w) > 3)
        if ramble_words:
            overlap = ramble_words & result_words
            ratio = len(overlap) / len(ramble_words)
            if ratio < 0.25:
                log.warning("Standup rewrote too much (overlap %.0f%%) — using raw", ratio * 100)
                return ramble
        log.info("Generated standup from %d words", len(ramble.split()))
        return result
    except Exception as e:
        log.error("Standup generation failed: %s", e)
        return ramble
