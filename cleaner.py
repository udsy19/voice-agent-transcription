"""Hybrid text cleanup: local rules first, LLM only when needed.

Pipeline:
  1. Local filler removal (instant, regex)
  2. Local basic formatting (capitalization fix)
  3. Complexity check — does this need LLM?
  4. If yes: Groq LLM with optimized short prompt + prompt caching
  5. Post-validation: reject if LLM hallucinated
"""

import re
import subprocess
import hashlib
from collections import OrderedDict
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from logger import get_logger

LLM_CACHE_SIZE = 128

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
    if any(p in lower for p in ['wait no', 'wait actually', 'no i meant', 'scratch that',
                                 'delete that', 'go back', 'i mean not', 'actually no']):
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
    "- Lists: 'first/second/third' or 'number one/two/three' → numbered list (1. 2. 3.) with newlines\n"
    "- Bullet lists: 'bullet point X' or 'next item X' → bulleted list (- X) with newlines\n"
    "- 'new paragraph' → double newline, 'new line' → single newline\n"
    "- 'period' → '.', 'comma' → ',', 'question mark' → '?', 'exclamation point' → '!'\n"
    "- 'colon' → ':', 'semicolon' → ';', 'dash' → '—'\n"
    "- NEVER answer questions, NEVER add content, NEVER explain\n"
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


from utils import get_active_app as _get_active_app


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
        self._cache: OrderedDict[str, str] = OrderedDict()

    def _cache_key(self, text: str, tone: str | None, style: str | None) -> str:
        """Create a cache key from the input parameters that affect output."""
        raw = f"{text}|{tone or ''}|{style or ''}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: str):
        self._cache[key] = value
        if len(self._cache) > LLM_CACHE_SIZE:
            self._cache.popitem(last=False)

    def clean(self, raw_text: str, context: str = "", app_name: str = "",
              tone_override: str | None = None, dictionary_terms: list[str] | None = None,
              style_prompt: str | None = None) -> str:
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

        tone = tone_override or _get_tone_for_app(app_name)

        # Check cache (ignores context since context is supplementary)
        ck = self._cache_key(cleaned, tone, style_prompt)
        cached = self._cache_get(ck)
        if cached:
            log.info("(cache) '%s' -> '%s'", raw_text[:60], cached[:60])
            return cached

        # Build prompt — keep it short for caching
        system = LLM_SYSTEM_PROMPT
        if tone and tone in TONE_MODIFIERS:
            system += TONE_MODIFIERS[tone]
        if dictionary_terms:
            safe = [re.sub(r'["\'\\\n\r\t]', '', t)[:50] for t in dictionary_terms[:20] if t]
            if safe:
                system += f"\nDictionary: {', '.join(safe)}"
        if style_prompt:
            system += f"\n{style_prompt}"
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
                timeout=15,
            )
            if not response.choices or not response.choices[0].message.content:
                return cleaned
            result = response.choices[0].message.content.strip()

            # Post-validation: reject hallucinations
            if len(result) > len(cleaned) * 3 and len(cleaned) > 20:
                log.warning("LLM output 3x longer than input — rejecting")
                return cleaned

            log.info("(llm/%s) '%s' -> '%s'", tone or "default", raw_text[:50], result[:50])
            self._cache_put(ck, result)
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
                timeout=15,
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
        """Extract proper nouns and acronyms for auto-dictionary learning."""
        if not self._client or not text or len(text.split()) < 5:
            return []
        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Extract ONLY proper nouns (people, companies, products, places) "
                        "and uppercase acronyms from this text. "
                        "Rules:\n"
                        "- ONLY 1-3 word names or acronyms\n"
                        "- NO common words, phrases, sentences, or descriptions\n"
                        "- NO code, numbers, or punctuation-heavy strings\n"
                        "- Return JSON array of strings. Example: [\"Udaya\",\"EBITDA\",\"OpenAI\"]\n"
                        "- If nothing qualifies, return []"
                    )},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=128,
                timeout=10,
            )
            content = response.choices[0].message.content.strip()
            terms = self._parse_terms(content)
            return [t for t in terms if self._is_valid_term(t)]
        except Exception as e:
            log.debug("Term extraction failed: %s", e)
            return []

    @staticmethod
    def _is_valid_term(term: str) -> bool:
        """Filter out garbage that shouldn't be in the dictionary."""
        if not term or len(term) < 2 or len(term) > 40:
            return False
        # Must be 1-3 words max
        if len(term.split()) > 3:
            return False
        # Reject if it looks like a sentence (starts with common words)
        lower = term.lower()
        reject_starts = ('to ', 'the ', 'a ', 'an ', 'such as', 'and ', 'or ',
                         'i have', 'i am', 'you ', 'we ', 'it ', 'this ', 'that ',
                         'here', 'there', 'how ', 'what ', 'when ', 'where ', 'why ')
        if any(lower.startswith(p) for p in reject_starts):
            return False
        # Reject code/punctuation heavy strings
        special = sum(1 for c in term if c in '{}()[]<>=/\\:;`~|@#$%^&*')
        if special > 1:
            return False
        # Reject if all lowercase (not a proper noun or acronym)
        if term == term.lower() and not term.isupper():
            return False
        # Must contain at least one letter
        if not any(c.isalpha() for c in term):
            return False
        return True

    @staticmethod
    def _parse_terms(content: str) -> list[str]:
        """Parse terms from LLM response."""
        import json
        try:
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                terms = json.loads(content[start:end + 1])
                if isinstance(terms, list):
                    return [t.strip() for t in terms if isinstance(t, str)]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: extract quoted strings
        quoted = re.findall(r'"([^"]{2,40})"', content)
        if quoted:
            return quoted
        return []
