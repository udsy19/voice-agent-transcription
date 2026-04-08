"""Unified LLM interface — local (mlx-lm) or cloud (Groq), switchable at runtime.

Usage:
    from llm import get_client
    client = get_client()
    response = client.chat(messages, model_tier="small")
    text = response.text

Model tiers:
    "small"  — fast, cheap: standup formatting, summaries, memory extraction
    "medium" — balanced: text cleanup, grammar correction
    "large"  — powerful: tool-calling assistant, complex reasoning
"""

import os
import json
import time
import threading
from logger import get_logger

log = get_logger("llm")

# ── Configuration ─────────────────────────────────────────────────────────

LOCAL_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
LOCAL_MODEL_FALLBACK = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # smaller fallback

# Groq model mapping by tier
GROQ_MODELS = {
    "small": "llama-3.1-8b-instant",
    "medium": "meta-llama/llama-4-scout-17b-16e-instruct",
    "large": "llama-3.3-70b-versatile",
}


# ── Response wrapper ──────────────────────────────────────────────────────

class ChatResponse:
    """Unified response from any LLM backend."""
    def __init__(self, text="", tool_calls=None, raw=None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw = raw  # original response object

    @property
    def has_tool_calls(self):
        return bool(self.tool_calls)


class ToolCall:
    """Represents a parsed tool call."""
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = type("Fn", (), {"name": name, "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments})()


# ── Base client ───────────────────────────────────────────────────────────

class LLMClient:
    """Base class for LLM backends."""
    def chat(self, messages, model_tier="small", tools=None, tool_choice=None,
             temperature=0.2, max_tokens=512, timeout=15):
        raise NotImplementedError


# ── Groq client ───────────────────────────────────────────────────────────

class GroqClient(LLMClient):
    """Cloud LLM via Groq API."""

    def __init__(self, api_key):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        log.info("Groq client initialized")

    def chat(self, messages, model_tier="small", tools=None, tool_choice=None,
             temperature=0.2, max_tokens=512, timeout=15):
        model = GROQ_MODELS.get(model_tier, GROQ_MODELS["small"])

        kwargs = {
            "model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "timeout": timeout,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        # Try primary model, fall back to 8b on rate limit
        for m in [model, GROQ_MODELS["small"]]:
            kwargs["model"] = m
            try:
                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                return ChatResponse(
                    text=choice.message.content or "",
                    tool_calls=choice.message.tool_calls,
                    raw=response,
                )
            except Exception as e:
                err = str(e)
                if ("429" in err or "rate_limit" in err) and m != GROQ_MODELS["small"]:
                    log.warning("%s rate limited, trying %s", m, GROQ_MODELS["small"])
                    continue
                raise

    @property
    def raw_client(self):
        """Access underlying Groq client for compound-beta, transcription, etc."""
        return self._client


# ── Local MLX client ──────────────────────────────────────────────────────

_local_model = None
_local_tokenizer = None
_local_lock = threading.Lock()


def _load_local_model():
    """Lazy-load the local MLX model."""
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer

    with _local_lock:
        if _local_model is not None:
            return _local_model, _local_tokenizer

        try:
            from mlx_lm import load
            t0 = time.time()
            log.info("Loading local LLM: %s ...", LOCAL_MODEL)
            _local_model, _local_tokenizer = load(LOCAL_MODEL)
            log.info("Local LLM loaded (%.1fs)", time.time() - t0)
            return _local_model, _local_tokenizer
        except Exception as e:
            log.warning("Failed to load %s: %s — trying fallback", LOCAL_MODEL, e)
            try:
                _local_model, _local_tokenizer = load(LOCAL_MODEL_FALLBACK)
                log.info("Fallback LLM loaded: %s", LOCAL_MODEL_FALLBACK)
                return _local_model, _local_tokenizer
            except Exception as e2:
                log.error("Local LLM unavailable: %s", e2)
                return None, None


class LocalClient(LLMClient):
    """Local LLM via mlx-lm on Apple Silicon."""

    def chat(self, messages, model_tier="small", tools=None, tool_choice=None,
             temperature=0.2, max_tokens=512, timeout=15):
        model, tokenizer = _load_local_model()
        if model is None:
            raise RuntimeError("Local LLM not available. Download models first.")

        from mlx_lm import generate

        # Build prompt from messages
        prompt = self._build_prompt(messages, tools)

        # Generate
        t0 = time.time()
        output = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, temp=temperature,
            verbose=False,
        )
        log.info("Local LLM (%.1fs, %d tokens): %s", time.time() - t0, len(output.split()), output[:60])

        # Parse tool calls from output if tools were provided
        if tools:
            tool_calls = self._parse_tool_calls(output)
            if tool_calls:
                return ChatResponse(text="", tool_calls=tool_calls)

        return ChatResponse(text=output.strip())

    def _build_prompt(self, messages, tools=None):
        """Build a chat prompt string from messages list."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n")
            elif role == "user":
                if tools and not any("Available tools:" in p for p in parts):
                    # Inject tool definitions into the first user message
                    tool_desc = self._format_tools(tools)
                    parts.append(f"[INST] {tool_desc}\n\n{content} [/INST]")
                else:
                    parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(content)
            elif role == "tool":
                parts.append(f"[Tool result: {content}]")
        return "\n".join(parts)

    def _format_tools(self, tools):
        """Format tool definitions for injection into prompt."""
        lines = ["Available tools (respond with JSON to call one):"]
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {}).get("properties", {})
            param_str = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in params.items())
            lines.append(f'- {name}({param_str}): {desc[:80]}')
        lines.append('\nTo call a tool, respond ONLY with: {"tool": "tool_name", "args": {...}}')
        lines.append("If no tool is needed, respond with plain text.")
        return "\n".join(lines)

    def _parse_tool_calls(self, output):
        """Parse tool calls from local model output."""
        import re
        output = output.strip()

        # Try to extract JSON object
        json_match = re.search(r'\{[\s\S]*\}', output)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        # Check for our tool call format: {"tool": "name", "args": {...}}
        if "tool" in data and isinstance(data.get("args"), dict):
            return [ToolCall(
                id=f"local_{int(time.time())}",
                name=data["tool"],
                arguments=data["args"],
            )]

        # Also try OpenAI-style: {"name": "...", "arguments": {...}}
        if "name" in data and "arguments" in data:
            return [ToolCall(
                id=f"local_{int(time.time())}",
                name=data["name"],
                arguments=data["arguments"],
            )]

        return []


# ── Hybrid client ─────────────────────────────────────────────────────────

class HybridClient(LLMClient):
    """Uses local for small/medium tasks, Groq for large/tool-calling."""

    def __init__(self, groq_key=None):
        self._groq = GroqClient(groq_key) if groq_key else None
        self._local = LocalClient()
        log.info("Hybrid client: groq=%s, local=mlx", "yes" if groq_key else "no")

    def chat(self, messages, model_tier="small", tools=None, tool_choice=None,
             temperature=0.2, max_tokens=512, timeout=15):
        # Route: tools or large tier → Groq (if available), else local
        use_groq = self._groq and (tools or model_tier == "large")

        if use_groq:
            try:
                return self._groq.chat(messages, model_tier, tools, tool_choice,
                                       temperature, max_tokens, timeout)
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate_limit" in err:
                    log.warning("Groq rate limited — falling back to local")
                else:
                    log.warning("Groq failed (%s) — falling back to local", str(e)[:60])

        # Local fallback
        try:
            return self._local.chat(messages, model_tier, tools, tool_choice,
                                    temperature, max_tokens, timeout)
        except Exception as e:
            # If local also fails and we haven't tried Groq yet, try it
            if self._groq and not use_groq:
                return self._groq.chat(messages, model_tier, tools, tool_choice,
                                       temperature, max_tokens, timeout)
            raise

    @property
    def raw_client(self):
        """Access underlying Groq client (for compound-beta, transcription)."""
        return self._groq.raw_client if self._groq else None


# ── Client singleton ──────────────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()


def get_client() -> LLMClient:
    """Get the global LLM client. Creates on first call based on config."""
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        from config import GROQ_API_KEY
        provider = os.getenv("LLM_PROVIDER", "hybrid")

        if provider == "groq" and GROQ_API_KEY:
            _client = GroqClient(GROQ_API_KEY)
        elif provider == "local":
            _client = LocalClient()
        else:
            _client = HybridClient(GROQ_API_KEY if GROQ_API_KEY else None)

        return _client


def get_raw_groq():
    """Get the raw Groq client for transcription/compound-beta. Returns None if no key."""
    client = get_client()
    if hasattr(client, 'raw_client'):
        return client.raw_client
    if isinstance(client, GroqClient):
        return client.raw_client
    return None
