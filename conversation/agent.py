"""Conversation Agent — optimized for low latency.

Pipeline: VAD → Transcribe + Screen + Memory (parallel) → Claude/Groq (streamed) → TTS (sentence-by-sentence)
Target: ~550ms from silence detection to first audio.
"""

import asyncio
import os
import re
import time
import numpy as np
import anthropic
from groq import Groq
from logger import get_logger
from transcriber import Transcriber
from conversation.vad_stream import VADStream
from conversation.context import ScreenContext
from conversation.tts import StreamingTTS
from conversation.memory import AgentMemory
from conversation.mcp_client import MCPClient
from conversation.desktop import DesktopController
from conversation.tools import TOOLS

log = get_logger("agent")

SYSTEM_PROMPT = """You are a voice AI agent running locally on the user's Mac. You are their personal assistant.

You can see their screen, hear them in real time, control their desktop, and remember things across conversations.

Rules:
- Respond in natural spoken language. No markdown, no bullet points, no code blocks.
- Be concise. 1-3 sentences unless more is needed. This is voice, not text.
- Use screen context to understand what the user is referring to.
- Prefer taking action directly over explaining how to do it.
- Ask for confirmation before: sending emails, deleting things, making purchases.
- When you execute a tool, briefly say what you're doing first."""

MODEL_VISION = "claude-sonnet-4-20250514"
MODEL_FAST = "llama-3.3-70b-versatile"  # Groq, no vision but 4x faster
MEMORY_EXTRACT_INTERVAL = 10

# Words that suggest user is referring to something on screen
VISION_WORDS = {"this", "that", "here", "screen", "what is", "what's", "look at",
                "see", "show", "showing", "displayed", "window", "page", "tab"}

SENTENCE_END = re.compile(r'(?<=[.!?])\s')

TOOL_FILLERS = [
    "Let me check that.",
    "On it.",
    "One moment.",
    "Looking into that.",
    "Sure, doing that now.",
]


def _needs_vision(transcript: str, last_app: str, current_app: str) -> bool:
    """Decide if this turn needs a screenshot (Claude) or can use fast Groq."""
    lower = transcript.lower()
    if any(w in lower for w in VISION_WORDS):
        return True
    if current_app != last_app and last_app:
        return True  # app changed, user might be referring to new content
    return False


class ConversationAgent:
    def __init__(self, emit_fn=None):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")

        if not api_key and not groq_key:
            raise ValueError("Need ANTHROPIC_API_KEY or GROQ_API_KEY in .env")

        self.claude = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.groq = Groq(api_key=groq_key) if groq_key else None
        self.vad = VADStream(stop_secs=0.4)  # fast VAD for conversation
        self.context = ScreenContext()
        self.tts = StreamingTTS()
        self.memory = AgentMemory()
        self.mcp = MCPClient()
        self.desktop = DesktopController()
        self.transcriber = Transcriber(backend="groq")
        self.active = False
        self._emit = emit_fn or (lambda msg: None)
        self._turn_count = 0
        self._last_app = ""
        self._filler_idx = 0

    async def start(self):
        self.active = True
        log.info("Conversation mode starting")
        self._emit({"type": "conversation", "status": "warming_up"})

        # Warmup in parallel
        await self._warmup()

        # Start MCP servers (non-blocking)
        asyncio.create_task(self._start_mcp())

        self._emit({"type": "conversation", "status": "listening"})
        log.info("Conversation mode ready")

        try:
            async for audio_chunk in self.vad.start():
                if not self.active:
                    break
                await self._handle_utterance(audio_chunk)
        except Exception as e:
            log.error("Conversation loop error: %s", e, exc_info=True)
        finally:
            await self.stop()

    async def _warmup(self):
        """Pre-warm all models so first response is fast."""
        t0 = time.time()
        tasks = []

        # Warm transcriber
        async def warm_transcriber():
            silent = np.zeros(1600, dtype=np.float32)
            self.transcriber.transcribe(silent)

        tasks.append(asyncio.to_thread(warm_transcriber))

        # Warm TTS (loads ONNX model)
        async def warm_tts():
            if self.tts._kokoro:
                self.tts._kokoro.create("warmup", voice=self.tts.voice, speed=1.0)

        tasks.append(warm_tts())

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass
        log.info("Warmup done in %.1fs", time.time() - t0)

    async def _start_mcp(self):
        try:
            await self.mcp.start_servers()
        except Exception as e:
            log.warning("MCP startup failed: %s", e)

    async def _handle_utterance(self, audio: np.ndarray):
        """Process one user utterance with maximum parallelism."""
        t0 = time.time()

        # Interrupt TTS if still playing
        if self.tts.is_playing:
            self.tts.interrupt()
            await asyncio.sleep(0.05)

        self._emit({"type": "conversation", "status": "transcribing"})

        # === PARALLEL: transcribe + capture screen + retrieve memory ===
        # Memory uses PREVIOUS turn's text (available instantly)
        prev_text = ""
        if self.memory.session_history:
            last = self.memory.session_history[-1]
            if isinstance(last.get("content"), str):
                prev_text = last["content"]

        transcript_future = asyncio.to_thread(self.transcriber.transcribe, audio)
        screen_future = asyncio.to_thread(self.context.capture)
        memory_future = asyncio.to_thread(self.memory.build_memory_block, prev_text)

        transcript, screen, memory_block = await asyncio.gather(
            transcript_future, screen_future, memory_future
        )

        if not transcript or len(transcript.strip()) < 2:
            self._emit({"type": "conversation", "status": "listening"})
            return

        log.info("User (%.0fms): %s", (time.time() - t0) * 1000, transcript)
        self._emit({"type": "conversation", "status": "thinking", "user_text": transcript})

        # Add to session
        self.memory.add_to_session("user", transcript)

        # === DECIDE: vision (Claude) or fast (Groq) ===
        current_app = screen.get("active_app", "")
        use_vision = _needs_vision(transcript, self._last_app, current_app)
        self._last_app = current_app

        # Build system prompt
        system = SYSTEM_PROMPT
        if memory_block:
            system += f"\n\nWHAT YOU REMEMBER:\n{memory_block}"
        if current_app:
            system += f"\n\nCURRENT APP: {current_app}"
        win = screen.get("window_title", "")
        if win:
            system += f" — {win}"

        # Build messages
        session_msgs = self.memory.get_session_messages()[:-1]

        if use_vision and self.claude and screen.get("screenshot_b64"):
            # Claude with vision
            user_content = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                              "data": screen["screenshot_b64"]}},
                {"type": "text", "text": transcript},
            ]
            await self._call_claude_streaming(system, session_msgs, user_content, t0)
        elif self.groq:
            # Groq fast path (no vision)
            await self._call_groq_streaming(system, session_msgs, transcript, t0)
        elif self.claude:
            # Claude without vision (fallback)
            user_content = [{"type": "text", "text": transcript}]
            await self._call_claude_streaming(system, session_msgs, user_content, t0)
        else:
            await self.tts.speak("I don't have an API key configured.")

        self._emit({"type": "conversation", "status": "listening"})

        # Periodic memory extraction (non-blocking)
        self._turn_count += 1
        if self._turn_count % MEMORY_EXTRACT_INTERVAL == 0 and self.claude:
            recent = self.memory.session_history[-MEMORY_EXTRACT_INTERVAL:]
            asyncio.create_task(self.memory.extract_and_store(recent, self.claude))

    async def _call_claude_streaming(self, system, history, user_content, t0):
        """Stream Claude response, speak sentence-by-sentence."""
        messages = history + [{"role": "user", "content": user_content}]
        mcp_tools = await self.mcp.list_tools()
        all_tools = TOOLS + mcp_tools

        try:
            with self.claude.messages.stream(
                model=MODEL_VISION,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=all_tools if all_tools else anthropic.NOT_GIVEN,
            ) as stream:
                await self._process_stream(stream, messages, all_tools, system, t0)
        except Exception as e:
            log.error("Claude error: %s", e)
            await self.tts.speak("Sorry, I had trouble with that.")

    async def _call_groq_streaming(self, system, history, transcript, t0):
        """Fast Groq path — no vision, streaming response."""
        messages = [{"role": "system", "content": system}]
        messages += history
        messages.append({"role": "user", "content": transcript})

        try:
            response = self.groq.chat.completions.create(
                model=MODEL_FAST,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                stream=True,
            )

            buffer = ""
            full_reply = ""
            first_sentence = True

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    buffer += delta.content
                    full_reply += delta.content

                    # Check for sentence end
                    parts = SENTENCE_END.split(buffer, 1)
                    if len(parts) > 1:
                        sentence = parts[0].strip()
                        buffer = parts[1]
                        if sentence:
                            if first_sentence:
                                latency = (time.time() - t0) * 1000
                                log.info("First sentence (%.0fms): %s", latency, sentence)
                                first_sentence = False
                            self._emit({"type": "conversation", "status": "speaking",
                                         "agent_text": sentence})
                            await self.tts.speak(sentence + ".")
                            if not self.active:
                                break

            # Speak remaining buffer
            if buffer.strip() and self.active:
                await self.tts.speak(buffer.strip())
                full_reply += ""

            if full_reply:
                self.memory.add_to_session("assistant", full_reply.strip())
                log.info("Agent (%.0fms total): %s", (time.time() - t0) * 1000, full_reply.strip()[:80])

        except Exception as e:
            log.error("Groq error: %s", e)
            await self.tts.speak("Sorry, I couldn't process that.")

    async def _process_stream(self, stream, messages, tools, system, t0):
        """Process Claude streaming response with tool handling."""
        buffer = ""
        full_reply = ""
        first_sentence = True

        response = stream.get_final_message()

        # Check for tool use
        if response.stop_reason == "tool_use":
            # Speak filler while executing tools
            filler = TOOL_FILLERS[self._filler_idx % len(TOOL_FILLERS)]
            self._filler_idx += 1
            asyncio.create_task(self.tts.speak(filler))

            # Execute tools
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    log.info("Tool: %s", block.name)
                    result = await self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            # Second Claude call with tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            try:
                follow_up = self.claude.messages.create(
                    model=MODEL_VISION,
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                    tools=tools if tools else anthropic.NOT_GIVEN,
                )
                reply = " ".join(b.text for b in follow_up.content if hasattr(b, "text")).strip()
                if reply:
                    # Wait for filler to finish
                    while self.tts.is_playing:
                        await asyncio.sleep(0.05)
                    self.memory.add_to_session("assistant", reply)
                    self._emit({"type": "conversation", "status": "speaking", "agent_text": reply})
                    await self.tts.speak(reply)
            except Exception as e:
                log.error("Follow-up Claude error: %s", e)
        else:
            # No tool use — extract text and speak
            reply = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
            if reply:
                self.memory.add_to_session("assistant", reply)
                latency = (time.time() - t0) * 1000
                log.info("Agent (%.0fms): %s", latency, reply[:80])
                self._emit({"type": "conversation", "status": "speaking", "agent_text": reply})

                # Speak sentence-by-sentence for perceived speed
                sentences = SENTENCE_END.split(reply)
                for s in sentences:
                    s = s.strip()
                    if s and self.active:
                        await self.tts.speak(s + ("." if not s[-1] in ".!?" else ""))

    async def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "type_text":
                return self.desktop.type_text(args["text"])
            elif name == "open_app":
                return self.desktop.open_app(args["app_name"])
            elif name == "press_shortcut":
                return self.desktop.press_shortcut(*args["keys"])
            elif name == "run_applescript":
                return self.desktop.run_applescript(args["script"])
            elif name == "query_screen_memory":
                return self.desktop.query_screenpipe(args["query"], args.get("minutes_back", 60))
            elif name == "click_at":
                return self.desktop.click(args["x"], args["y"])
            elif "__" in name:
                server, tool = name.split("__", 1)
                return await self.mcp.call_tool(server, tool, args)
            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {e}"

    async def stop(self):
        self.active = False
        self.tts.interrupt()
        await self.vad.stop()
        await self.mcp.stop_servers()
        self._emit({"type": "conversation", "status": "stopped"})
        log.info("Conversation mode stopped")
