"""Conversation Agent — production-grade with all optimizations.

3-tier routing → parallel context fetch → structured tool execution → streaming TTS
Speculative execution, dialogue state tracking, user profile, graph memory, metrics.
"""

import asyncio
import os
import re
import time
import random
import numpy as np
import anthropic
from groq import Groq
from logger import get_logger
from transcriber import Transcriber
from conversation.context import ScreenContext
from conversation.tts import StreamingTTS
from conversation.memory import AgentMemory
from conversation.mcp_client import MCPClient
from conversation.desktop import DesktopController
from conversation.executor import ToolExecutor
from conversation.tools import TOOLS, TOOL_FILLERS
from conversation.router import route, build_app_context, get_installed_apps
from conversation.metrics import METRICS

log = get_logger("agent")

SENTENCE_END = re.compile(r'(?<=[.!?])\s')

SYSTEM_PROMPT = """You are an autonomous desktop agent running locally on the user's Mac.

When given a task, you:
1. Break it into steps
2. Execute each step using the fastest available tool
3. Verify the result before moving on
4. Speak a brief status update after major steps

TOOL PRIORITY (fastest to slowest):
1. AppleScript — for any native macOS app (Mail, Calendar, Finder, Safari, Messages)
2. read_app_ui → click_element — for any app with standard UI
3. MCP tools — for Gmail, Google Calendar, filesystem
4. query_screen_memory — for retrieving past context
5. click_at_coordinates — absolute last resort

RULES:
- Respond in natural spoken language. No markdown, no formatting, no asterisks.
- Be concise. 1-3 sentences. This is voice, not text.
- NEVER make up data. If you don't know, use a tool to find out.
- NEVER hallucinate names, emails, messages, or events.
- Never ask permission for read operations or reversible actions
- Always ask before: sending emails/messages, deleting, purchases
- While executing tools, briefly tell the user what you're doing
- If a tool fails, try the next method in priority order
- Maximum 5 tool calls before asking user for guidance"""

MODEL_VISION = "claude-sonnet-4-20250514"
MODEL_FAST = "llama-3.3-70b-versatile"
MEMORY_EXTRACT_INTERVAL = 10


class ConversationAgent:
    def __init__(self, emit_fn=None):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")

        if not api_key and not groq_key:
            raise ValueError("Need ANTHROPIC_API_KEY or GROQ_API_KEY in .env")

        self.claude = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.groq = Groq(api_key=groq_key) if groq_key else None
        self.context = ScreenContext()
        self.tts = StreamingTTS()
        self.memory = AgentMemory()
        self.mcp = MCPClient()
        self.desktop = DesktopController()
        self.executor = ToolExecutor(self.desktop, METRICS)
        self.transcriber = Transcriber(backend="groq" if groq_key else "faster-whisper")
        self.active = False
        self._emit = emit_fn or (lambda msg: None)
        self._turn_count = 0
        self._app_context = ""

    async def start(self):
        self.active = True
        self._emit({"type": "conversation", "status": "warming_up"})
        asyncio.create_task(self.tts.speak("Setting up."))

        await asyncio.gather(
            self._warmup(),
            self._start_mcp(),
            return_exceptions=True,
        )

        self._app_context = build_app_context()
        self._emit({"type": "conversation", "status": "listening"})
        log.info("Ready. Apps: %s", get_installed_apps())
        await self.tts.speak("Ready. Hold right option and speak.")

    async def _warmup(self):
        t0 = time.time()
        try:
            silent = np.zeros(1600, dtype=np.float32)
            await asyncio.to_thread(self.transcriber.transcribe, silent)
        except Exception:
            pass
        METRICS.record("warmup_ms", (time.time() - t0) * 1000)

    async def _start_mcp(self):
        try:
            await self.mcp.start_servers()
        except Exception as e:
            log.warning("MCP: %s", e)

    async def process_audio(self, audio: np.ndarray):
        """Process one user utterance."""
        if not self.active:
            return
        t0 = time.time()

        if self.tts.is_playing:
            self.tts.interrupt()
            await asyncio.sleep(0.05)

        self._emit({"type": "conversation", "status": "transcribing"})

        # Parallel: transcribe + screen + memory
        prev = ""
        if self.memory.session_history:
            last = self.memory.session_history[-1]
            if isinstance(last.get("content"), str):
                prev = last["content"]

        transcript, screen, mem_block = await asyncio.gather(
            asyncio.to_thread(self.transcriber.transcribe, audio),
            asyncio.to_thread(self.context.capture),
            asyncio.to_thread(self.memory.build_memory_block, prev),
        )

        METRICS.record("transcription_ms", (time.time() - t0) * 1000)

        if not transcript or len(transcript.strip()) < 2:
            self._emit({"type": "conversation", "status": "listening"})
            return

        log.info("User (%.0fms): %s", (time.time() - t0) * 1000, transcript)
        self._emit({"type": "conversation", "status": "thinking", "user_text": transcript})
        self.memory.add_to_session("user", transcript)

        # Route
        tier, instant_action = route(transcript)
        current_app = screen.get("active_app", "")
        self.memory.dialogue.current_app = current_app

        try:
            if tier == "instant" and instant_action:
                await self._handle_instant(instant_action, t0)
            elif tier == "groq" and self.groq:
                await self._handle_groq(transcript, screen, mem_block, t0)
            elif self.claude:
                await self._handle_claude(transcript, screen, mem_block, t0)
            elif self.groq:
                await self._handle_groq(transcript, screen, mem_block, t0)
            else:
                await self.tts.speak("No API key configured.")
        except Exception as e:
            log.error("Processing error: %s", e, exc_info=True)
            await self.tts.speak("Sorry, something went wrong. Try again.")

        self._emit({"type": "conversation", "status": "listening"})
        METRICS.record("total_turn_ms", (time.time() - t0) * 1000)

        # Memory extraction
        self._turn_count += 1
        if self._turn_count % MEMORY_EXTRACT_INTERVAL == 0 and self.claude:
            asyncio.create_task(
                self.memory.extract_and_store(
                    self.memory.session_history[-MEMORY_EXTRACT_INTERVAL:], self.claude
                )
            )

    async def _handle_instant(self, action, t0):
        result = await self.desktop.execute_instant(action)
        narration = self._narrate(action, result)
        self._emit({"type": "conversation", "status": "speaking", "agent_text": narration})
        await self.tts.speak(narration)
        self.memory.add_to_session("assistant", narration)
        self.memory.dialogue.update(narration, action.get("action", ""))
        METRICS.record("instant_ms", (time.time() - t0) * 1000)

    async def _handle_groq(self, transcript, screen, mem_block, t0):
        system = self._build_system(screen, mem_block)
        messages = [{"role": "system", "content": system}]
        messages += self.memory.get_session_messages()[:-1]
        messages.append({"role": "user", "content": transcript})

        try:
            stream = self.groq.chat.completions.create(
                model=MODEL_FAST, messages=messages,
                max_tokens=512, temperature=0.7, stream=True,
            )
            full = await self._stream_and_speak(stream, t0)
            if full:
                self.memory.add_to_session("assistant", full)
                self.memory.dialogue.update(transcript, "responded")
        except Exception as e:
            log.error("Groq: %s", e)
            await self.tts.speak("Sorry, I couldn't process that.")

    async def _handle_claude(self, transcript, screen, mem_block, t0):
        system = self._build_system(screen, mem_block)
        user_content = []
        if screen.get("screenshot_b64"):
            user_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                            "data": screen["screenshot_b64"]},
            })
        user_content.append({"type": "text", "text": transcript})

        messages = self.memory.get_session_messages()[:-1]
        messages.append({"role": "user", "content": user_content})

        mcp_tools = await self.mcp.list_tools()
        all_tools = TOOLS + mcp_tools

        for iteration in range(5):
            try:
                response = self.claude.messages.create(
                    model=MODEL_VISION, max_tokens=1024,
                    system=system, messages=messages,
                    tools=all_tools if all_tools else anthropic.NOT_GIVEN,
                )
            except Exception as e:
                log.error("Claude: %s", e)
                await self.tts.speak("Sorry, I had trouble with that.")
                return

            if response.stop_reason == "tool_use":
                first_tool = next((b for b in response.content if b.type == "tool_use"), None)
                if first_tool:
                    fillers = TOOL_FILLERS.get(first_tool.name, ["On it..."])
                    asyncio.create_task(self.tts.speak(random.choice(fillers)))

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        log.info("Tool [%d]: %s", iteration, block.name)
                        result = await self.executor.execute(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result.result[:2000] if result.ok else f"Error: {result.error}",
                        })
                        # Update dialogue state
                        self.memory.dialogue.update(
                            block.name, f"{block.name}({list(block.input.keys())})",
                            entities=[str(v)[:50] for v in block.input.values()],
                        )

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                METRICS.record("tool_call_ms", (time.time() - t0) * 1000, {"tool": first_tool.name if first_tool else ""})
                continue

            # End turn — speak
            reply = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
            if reply:
                while self.tts.is_playing:
                    await asyncio.sleep(0.05)
                self.memory.add_to_session("assistant", reply)
                self.memory.dialogue.update(reply, "responded")
                self._emit({"type": "conversation", "status": "speaking", "agent_text": reply})
                await self._speak_sentences(reply)
                METRICS.record("claude_turn_ms", (time.time() - t0) * 1000)
            break

    async def _stream_and_speak(self, stream, t0) -> str:
        """Stream Groq response, speak sentence by sentence."""
        buffer = ""
        full = ""
        first = True

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                buffer += delta.content
                full += delta.content
                parts = SENTENCE_END.split(buffer, 1)
                if len(parts) > 1:
                    sentence = parts[0].strip()
                    buffer = parts[1]
                    if sentence:
                        if first:
                            METRICS.record("ttfs_ms", (time.time() - t0) * 1000)
                            log.info("TTFS (%.0fms): %s", (time.time() - t0) * 1000, sentence[:50])
                            first = False
                        self._emit({"type": "conversation", "status": "speaking", "agent_text": sentence})
                        await self.tts.speak(sentence + ("." if sentence[-1] not in ".!?" else ""))
                        if not self.active:
                            break

        if buffer.strip() and self.active:
            await self.tts.speak(buffer.strip())
        return full.strip()

    async def _speak_sentences(self, text: str):
        """Speak text sentence by sentence for responsiveness."""
        sentences = SENTENCE_END.split(text)
        for s in sentences:
            s = s.strip()
            if s and self.active:
                await self.tts.speak(s + ("." if s[-1] not in ".!?" else ""))

    def _narrate(self, action: dict, result: str) -> str:
        act = action.get("action", "")
        narrations = {
            "open_app": f"Opening {action.get('app', 'app')}.",
            "type": "Typed.", "undo": "Undone.",
            "screenshot": "Screenshot taken.", "media": "Done.",
            "volume": f"Volume {action.get('direction', 'adjusted')}.",
            "lock": "Screen locked.",
        }
        return narrations.get(act, "Done.")

    def _build_system(self, screen, mem_block) -> str:
        parts = [SYSTEM_PROMPT]
        if mem_block:
            parts.append(f"\n{mem_block}")
        if self._app_context:
            parts.append(f"\n{self._app_context}")
        app = screen.get("active_app", "")
        win = screen.get("window_title", "")
        if app:
            parts.append(f"\nCURRENT: {app}" + (f" — {win}" if win else ""))
        return "\n".join(parts)

    async def stop(self):
        self.active = False
        self.tts.interrupt()
        self.memory.save_session()
        self.memory.save_profile()
        await self.mcp.stop_servers()
        METRICS.close()
        self._emit({"type": "conversation", "status": "stopped"})
        log.info("Conversation stopped")
