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

        # Try primary model, fall back to 8b on rate limit or tool-call failure
        for m in [model, GROQ_MODELS["small"]]:
            kwargs["model"] = m
            _t0 = time.time()
            try:
                response = self._client.chat.completions.create(**kwargs)
                try:
                    import robustness as _rb
                    _rb.record_groq_call()
                    _rb.record_metric("groq_chat", (time.time() - _t0) * 1000)
                except Exception:
                    pass
                choice = response.choices[0]
                return ChatResponse(
                    text=choice.message.content or "",
                    tool_calls=choice.message.tool_calls,
                    raw=response,
                )
            except Exception as e:
                err = str(e)
                # Rate limit → try smaller model
                if ("429" in err or "rate_limit" in err) and m != GROQ_MODELS["small"]:
                    log.warning("%s rate limited, trying %s", m, GROQ_MODELS["small"])
                    continue
                # Tool call format failure → retry without tool_choice=auto (let model decide)
                if tools and "Failed to call a function" in err and m == model:
                    log.warning("%s tool call failed, retrying with different model", m)
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
        from mlx_lm.sample_utils import make_sampler

        # Build prompt from messages
        prompt = self._build_prompt(messages, tools)

        # Tool calls require deterministic output — force low temperature
        effective_temp = 0.01 if tools else max(temperature, 0.01)

        # Generate with sampler for temperature control
        t0 = time.time()
        sampler = make_sampler(temp=effective_temp)
        try:
            output = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=min(max_tokens, 1024),  # cap to avoid slow generation
                sampler=sampler,
                verbose=False,
            )
        except TypeError:
            # Fallback: older mlx-lm versions without sampler param
            output = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=min(max_tokens, 1024),
                verbose=False,
            )
        duration = time.time() - t0
        try:
            import robustness as _rb
            _rb.record_metric("local_chat", duration * 1000)
        except Exception:
            pass
        log.info("Local LLM (%.1fs): %s", duration, output[:80])

        # Parse tool calls from output if tools were provided
        if tools:
            tool_calls = self._parse_tool_calls(output)
            if tool_calls:
                return ChatResponse(text="", tool_calls=tool_calls)
            # If tools were required but local LLM didn't produce any, this is unreliable
            # Raise so the hybrid client can surface a clear error rather than hallucinate
            raise RuntimeError("Local LLM cannot reliably use tools — requires cloud LLM")

        return ChatResponse(text=output.strip())

    def _build_prompt(self, messages, tools=None):
        """Build a chat prompt from messages. Uses tokenizer template if available."""
        model, tokenizer = _load_local_model()

        # Inject tool definitions into the system message if tools provided
        if tools:
            tool_desc = self._format_tools(tools)
            # Prepend to first system message, or add as system message
            injected = False
            for msg in messages:
                if msg.get("role") == "system" and not injected:
                    msg = dict(msg)  # don't mutate original
                    msg["content"] = msg["content"] + "\n\n" + tool_desc
                    injected = True
                    break
            if not injected:
                messages = [{"role": "system", "content": tool_desc}] + list(messages)

        # Try tokenizer's chat template (Mistral v0.3 has one)
        try:
            if hasattr(tokenizer, 'apply_chat_template'):
                # Filter out tool role messages — not all templates support them
                clean_msgs = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "tool":
                        clean_msgs.append({"role": "user", "content": f"[Tool result]: {content}"})
                    elif role in ("system", "user", "assistant"):
                        clean_msgs.append({"role": role, "content": content})
                return tokenizer.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            log.debug("Chat template failed: %s — using manual format", e)

        # Manual fallback (Mistral format)
        parts = []
        sys_content = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                sys_content = content
            elif role == "user":
                prefix = f"{sys_content}\n\n" if sys_content else ""
                sys_content = ""
                parts.append(f"[INST] {prefix}{content} [/INST]")
            elif role == "assistant":
                parts.append(content + "</s>")
            elif role == "tool":
                parts.append(f"[INST] Tool result: {content} [/INST]")
        return "\n".join(parts)

    def _format_tools(self, tools):
        """Format tool definitions for local LLM prompt injection.

        Mirrors OpenAI/Groq function-calling semantics so the prompt layer is
        identical across backends — only the delivery differs.
        """
        lines = ["You have access to the following tools:"]
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            props = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            param_lines = []
            for k, v in props.items():
                req = " (required)" if k in required else ""
                param_lines.append(f'    "{k}": {v.get("type", "string")}{req} — {v.get("description", "")[:60]}')
            params_desc = "\n".join(param_lines) if param_lines else "    (no parameters)"
            lines.append(f'\n- {name}: {desc[:120]}\n  Parameters:\n{params_desc}')
        lines.append('\n\nWhen a tool is needed, respond with ONLY this exact JSON object (no other text, no markdown):')
        lines.append('{"tool": "<tool_name>", "args": {"<param>": "<value>"}}')
        lines.append('\nExample — user asks "what\'s on my calendar today":')
        lines.append('{"tool": "list_calendar_events", "args": {"days": "1"}}')
        lines.append('\nIf no tool is needed, respond with a brief helpful answer in plain text.')
        lines.append('Never explain the tool call. Never wrap in markdown. Output the raw JSON only.')
        return "\n".join(lines)

    def _parse_tool_calls(self, output):
        """Parse tool calls from local model output — robust to extra text/markdown."""
        import re
        text = output.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            fence = re.match(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if fence:
                text = fence.group(1).strip()

        # Collect candidate JSON objects — scan for balanced-brace spans
        candidates = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start:i + 1])
                    start = -1

        # Try each candidate, return first that matches our tool call shape
        for cand in candidates:
            try:
                data = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            # Shape 1: {"tool": "name", "args": {...}}
            if "tool" in data and isinstance(data.get("args"), dict):
                return [ToolCall(id=f"local_{int(time.time()*1000)}",
                                 name=str(data["tool"]), arguments=data["args"])]
            # Shape 2: OpenAI-style {"name": "...", "arguments": {...} or "..." }
            if "name" in data and "arguments" in data:
                args = data["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
                if isinstance(args, dict):
                    return [ToolCall(id=f"local_{int(time.time()*1000)}",
                                     name=str(data["name"]), arguments=args)]
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
        # Routing:
        # - Tools always prefer Groq when available (most reliable tool calling)
        # - Large tier prefers Groq (better quality for complex reasoning)
        # - Small/medium without tools: respect LLM_PROVIDER setting
        provider = os.getenv("LLM_PROVIDER", "hybrid")
        prefer_groq = self._groq and (tools or model_tier == "large" or provider == "groq")

        if prefer_groq:
            try:
                return self._groq.chat(messages, model_tier, tools, tool_choice,
                                       temperature, max_tokens, timeout)
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate_limit" in err:
                    log.warning("Groq rate limited — falling back to local")
                else:
                    log.warning("Groq failed (%s) — falling back to local", str(e)[:60])

        # Local path (or Groq fallback for non-tool calls)
        try:
            return self._local.chat(messages, model_tier, tools, tool_choice,
                                    temperature, max_tokens, timeout)
        except Exception as e:
            # If tools required and local failed, try Groq one more time before giving up
            if tools and self._groq and not prefer_groq:
                try:
                    return self._groq.chat(messages, model_tier, tools, tool_choice,
                                           temperature, max_tokens, timeout)
                except Exception:
                    pass
            if tools:
                return ChatResponse(
                    text="I need to look that up but I can't reach the cloud service right now. Check your connection or Groq API key.",
                )
            if self._groq and not prefer_groq:
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
