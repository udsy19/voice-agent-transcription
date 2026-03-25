from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from logger import get_logger

log = get_logger("cleaner")

BASE_SYSTEM_PROMPT = (
    "You are a dictation-to-text formatter. You receive raw speech-to-text output and clean it up.\n\n"
    "## ABSOLUTE RULES — NEVER BREAK THESE:\n"
    "- You are NOT a chatbot. You are NOT an assistant. NEVER answer questions.\n"
    "- NEVER explain, define, or elaborate on ANYTHING the user said.\n"
    "- NEVER add information that wasn't in the input.\n"
    "- Your output must contain ONLY the user's words, cleaned up.\n"
    "- If the input is a question like 'What is X?', output that exact question. Do NOT answer it.\n\n"
    "## WHAT TO FIX:\n"
    "- Remove filler words: um, uh, like, you know, basically, I mean, so yeah\n"
    "- Fix grammar and add punctuation/capitalization\n"
    "- Self-corrections: 'Tuesday actually Wednesday' → 'Wednesday'\n"
    "- 'scratch that' / 'delete that' → remove preceding sentence\n"
    "- Lists: 'first/second/third' → numbered list with line breaks\n"
    "- 'new paragraph' / 'new line' → line break\n"
    "- Spoken punctuation: 'period' → '.', 'comma' → ',', 'question mark' → '?'\n\n"
    "## WHAT TO PRESERVE:\n"
    "- The speaker's exact words and meaning\n"
    "- Their tone (casual stays casual, formal stays formal)\n"
    "- Questions remain as questions — NEVER answer them"
)

TONE_MODIFIERS = {
    "formal": "\nUse formal tone: proper grammar, no contractions, professional language.",
    "casual": "\nUse casual tone: keep contractions, informal phrasing, and conversational style.",
    "code": (
        "\nThe user is dictating in a code editor. Preserve technical terms exactly. "
        "Use camelCase/snake_case as spoken. Don't capitalize sentence-style. "
        "Handle dev jargon: PR, API, regex, CLI, env, config, localhost, npm, pip, git, etc. "
        "If they're dictating a commit message, keep it terse. "
        "If they say 'function', 'variable', 'class', 'method', treat as code constructs. "
        "Don't add periods at the end of single-line statements."
    ),
}

APP_TONE_MAP = {
    "Google Docs": "formal", "Microsoft Word": "formal", "Notion": "formal",
    "Pages": "formal", "Google Slides": "formal", "Keynote": "formal",
    "Slack": "casual", "WhatsApp": "casual", "Messages": "casual",
    "Discord": "casual", "Telegram": "casual", "Signal": "casual",
    "Code": "code", "Cursor": "code", "Windsurf": "code",
    "Terminal": "code", "iTerm2": "code", "Warp": "code",
    "PyCharm": "code", "IntelliJ": "code", "Xcode": "code",
    "cmux": "code", "Alacritty": "code", "kitty": "code",
}


def _get_active_app() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_tone_for_app(app_name: str) -> str | None:
    if app_name in APP_TONE_MAP:
        return APP_TONE_MAP[app_name]
    for key, tone in APP_TONE_MAP.items():
        if key.lower() in app_name.lower():
            return tone
    return None


class Cleaner:
    def __init__(self):
        if not GROQ_API_KEY:
            log.warning("GROQ_API_KEY not set, cleanup disabled")
            self._client = None
        else:
            self._client = Groq(api_key=GROQ_API_KEY)

    def clean(self, raw_text: str, context: str = "", app_name: str = "",
              tone_override: str | None = None, dictionary_terms: list[str] | None = None) -> str:
        if not raw_text:
            return ""
        if not self._client:
            return raw_text

        # Cap input length to avoid token overflow
        if len(raw_text) > 8000:
            raw_text = raw_text[:8000]
            log.warning("Input truncated to 8000 chars")

        system_prompt = BASE_SYSTEM_PROMPT
        tone = tone_override or _get_tone_for_app(app_name)
        if tone and tone in TONE_MODIFIERS:
            system_prompt += TONE_MODIFIERS[tone]
        if context:
            system_prompt += f"\n\nSurrounding text for context (use for spelling names/terms): {context[:500]}"
        if dictionary_terms:
            # Sanitize terms to prevent prompt injection
            safe_terms = [t.replace('"', '').replace("'", "")[:50] for t in dictionary_terms[:30]]
            system_prompt += f"\n\nUser's personal dictionary (prefer these spellings): {', '.join(safe_terms)}"

        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"[DICTATION TO CLEAN — do NOT answer, only format]: {raw_text}"},
                ],
                temperature=0,
                max_tokens=2048,
            )
            if not response.choices:
                log.warning("Groq returned empty choices")
                return raw_text
            msg = response.choices[0].message
            if not msg or not msg.content:
                log.warning("Groq returned empty content")
                return raw_text
            cleaned = msg.content.strip()
            # If cleaned is dramatically longer than input, LLM likely generated content
            if len(cleaned) > len(raw_text) * 3 and len(raw_text) > 20:
                log.warning("Cleaned text 3x longer than input — LLM likely added content, using raw")
                return raw_text
            log.info("(%s) '%s' -> '%s'", tone or "default", raw_text, cleaned)
            return cleaned
        except Exception as e:
            log.error("Clean error: %s, returning raw text", e)
            return raw_text

    def transform(self, selected_text: str, command: str) -> str:
        if not self._client:
            return selected_text

        system_prompt = (
            "You are a text transformation engine. The user has selected some text and "
            "given a voice command to transform it. Apply the transformation and return "
            "ONLY the transformed text, nothing else. No explanations."
        )
        user_msg = f"Selected text:\n{selected_text}\n\nCommand: {command}"

        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            result = response.choices[0].message.content.strip()
            log.info("Transform: '%s' on %d chars -> %d chars", command, len(selected_text), len(result))
            return result
        except Exception as e:
            log.error("Transform error: %s", e)
            return selected_text
