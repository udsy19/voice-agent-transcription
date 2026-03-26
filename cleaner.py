"""Hybrid text cleanup: local rules first, LLM only when needed.

Pipeline:
  1. Local filler removal (instant, regex)
  2. Local basic formatting (capitalization fix)
  3. Complexity check — does this need LLM?
  4. If yes: Groq LLM with optimized short prompt + prompt caching
  5. Post-validation: reject if LLM hallucinated
"""

import json
import re
import subprocess
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from logger import get_logger

log = get_logger("cleaner")

# ── Local filler removal (runs on every request, ~0ms) ──────────────────────

FILLER_PATTERN = re.compile(
    r'\b('
    r'um+|uh+|er+|ah+|hmm+|hm+|'
    r'like\s*,?\s*(?=\w)|'  # "like" as filler (not "I like")
    r'you know\s*,?\s*|'
    r'I mean\s*,?\s*|'
    r'basically\s*,?\s*|'
    r'actually\s*,?\s*(?=,)|'  # "actually," (filler only when followed by comma)
    r'sort of\s*,?\s*|'
    r'kind of\s*,?\s*|'
    r'right\s*,?\s*(?=so|and|but|the|we|I|you)|'  # "right, so..."
    r'so yeah\s*,?\s*|'
    r'yeah\s*,?\s*(?=so|and|but|the|we|I)'  # "yeah, so..."
    r')\s*',
    re.IGNORECASE,
)

REPEATED_SPACES = re.compile(r' {2,}')
REPEATED_COMMAS = re.compile(r',\s*,')


def _local_clean(text: str) -> str:
    """Fast local cleanup: filler removal + basic formatting."""
    text = FILLER_PATTERN.sub(' ', text)
    text = REPEATED_SPACES.sub(' ', text)
    text = REPEATED_COMMAS.sub(',', text)
    text = text.strip()
    # Fix leading lowercase after filler removal
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _needs_llm(raw: str, cleaned: str) -> bool:
    """Decide if text needs LLM processing beyond local cleanup."""
    lower = cleaned.lower()
    # Backtracking patterns
    if any(p in lower for p in ['actually', 'wait', 'no i meant', 'scratch that',
                                 'delete that', 'go back', 'i mean not']):
        return True
    # List patterns
    if any(p in lower for p in ['first', 'second', 'third', 'number one',
                                 'number two', 'bullet', 'next item']):
        return True
    # Formatting commands
    if any(p in lower for p in ['new paragraph', 'new line', 'period', 'comma',
                                 'question mark', 'exclamation point']):
        return True
    # Long text benefits from grammar cleanup
    if len(cleaned.split()) > 20:
        return True
    # Tone/style adaptation always needs LLM
    return False


# ── LLM System Prompt (kept SHORT for prompt caching + speed) ───────────────

# This exact prefix gets cached by Groq after first request = 50% cost savings
LLM_SYSTEM_PROMPT = (
    "Clean this dictation. Rules:\n"
    "- Fix grammar, punctuation, capitalization\n"
    "- Self-corrections: 'X actually Y' → keep only Y\n"
    "- 'scratch that'/'delete that' → remove preceding sentence\n"
    "- Lists: 'first/second/third' → numbered list with newlines\n"
    "- 'new paragraph' → line break, 'period' → '.'\n"
    "- NEVER answer questions, NEVER add content\n"
    "Output ONLY the cleaned text."
)

TONE_MODIFIERS = {
    "formal": " Use formal tone, no contractions.",
    "casual": " Keep casual tone, contractions OK.",
    "code": " Code context: preserve technical terms, camelCase/snake_case, no periods on single lines.",
}

APP_TONE_MAP = {
    "Google Docs": "formal", "Microsoft Word": "formal", "Notion": "formal",
    "Pages": "formal", "Keynote": "formal",
    "Slack": "casual", "WhatsApp": "casual", "Messages": "casual",
    "Discord": "casual", "Telegram": "casual", "Signal": "casual",
    "Code": "code", "Cursor": "code", "Windsurf": "code",
    "Terminal": "code", "iTerm2": "code", "Warp": "code",
    "PyCharm": "code", "IntelliJ": "code", "Xcode": "code",
    "cmux": "code", "Alacritty": "code", "kitty": "code",
}


def _get_active_app() -> str:
    try:
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
            log.warning("GROQ_API_KEY not set, LLM cleanup disabled")
            self._client = None
        else:
            self._client = Groq(api_key=GROQ_API_KEY)

    def clean(self, raw_text: str, context: str = "", app_name: str = "",
              tone_override: str | None = None, dictionary_terms: list[str] | None = None) -> str:
        if not raw_text:
            return ""

        # Step 1: Local cleanup (instant)
        cleaned = _local_clean(raw_text)
        if not cleaned:
            return ""

        # Step 2: Check if LLM is needed
        if not self._client or not _needs_llm(raw_text, cleaned):
            log.info("(local) '%s' -> '%s'", raw_text[:60], cleaned[:60])
            return cleaned

        # Step 3: LLM cleanup for complex cases
        if len(cleaned) > 8000:
            cleaned = cleaned[:8000]

        # Build prompt — keep it short for caching
        system = LLM_SYSTEM_PROMPT
        tone = tone_override or _get_tone_for_app(app_name)
        if tone and tone in TONE_MODIFIERS:
            system += TONE_MODIFIERS[tone]
        if dictionary_terms:
            safe = [t.replace('"', '').replace("'", "")[:50] for t in dictionary_terms[:20]]
            system += f"\nDictionary: {', '.join(safe)}"
        if context:
            system += f"\nContext: {context[:300]}"

        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"[DICTATION]: {cleaned}"},
                ],
                temperature=0,
                max_tokens=2048,
            )
            if not response.choices or not response.choices[0].message.content:
                return cleaned
            result = response.choices[0].message.content.strip()

            # Post-validation: reject hallucinations
            if len(result) > len(cleaned) * 3 and len(cleaned) > 20:
                log.warning("LLM output 3x longer than input — rejecting")
                return cleaned

            log.info("(llm/%s) '%s' -> '%s'", tone or "default", raw_text[:50], result[:50])
            return result
        except Exception as e:
            log.error("LLM error: %s, returning local cleanup", e)
            return cleaned

    def transform(self, selected_text: str, command: str) -> str:
        """Command mode: transform selected text."""
        if not self._client:
            return selected_text
        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Transform the text as instructed. Return ONLY the result."},
                    {"role": "user", "content": f"Text:\n{selected_text}\n\nInstruction: {command}"},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            if not response.choices or not response.choices[0].message.content:
                return selected_text
            result = response.choices[0].message.content.strip()
            log.info("Transform: '%s' -> %d chars", command[:40], len(result))
            return result
        except Exception as e:
            log.error("Transform error: %s", e)
            return selected_text

    def extract_terms(self, text: str) -> list[str]:
        """Extract proper nouns, emails, acronyms for auto-dictionary learning."""
        if not self._client or not text or len(text) < 10:
            return []
        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Extract notable terms: proper nouns, emails, acronyms, jargon. "
                        "Return JSON array. Example: [\"Udaya\",\"test@email.com\",\"EBITDA\"] "
                        "If none, return []"
                    )},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            if content.startswith("["):
                terms = json.loads(content)
                if isinstance(terms, list):
                    return [t for t in terms if isinstance(t, str) and len(t) > 1]
            return []
        except Exception as e:
            log.debug("Term extraction failed: %s", e)
            return []
