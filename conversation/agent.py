"""Conversation Agent — walkie-talkie style.

Hold Right Option → speak → release → Claude responds via TTS.
Same hotkey as dictation, but routes to Claude instead of paste.
Uses 3-tier routing + agentic execution loop.
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
from conversation.tools import TOOLS, TOOL_FILLERS
from conversation.router import route, build_app_context, get_installed_apps

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
- Respond in natural spoken language. No markdown, no bullet points.
- Be concise. 1-3 sentences. This is voice.
- Never ask permission for read operations or reversible actions
- Always ask before: sending emails/messages, deleting, purchases, calendar changes
- If a step fails, try the next tool in priority order
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
        self.transcriber = Transcriber(backend="groq" if groq_key else "faster-whisper")
        self.active = False
        self._emit = emit_fn or (lambda msg: None)
        self._turn_count = 0
        self._last_app = ""
        self._app_context = ""

    async def start(self):
        """Initialize conversation mode. Audio is pushed via process_audio()."""
        self.active = True
        self._emit({"type": "conversation", "status": "warming_up"})

        await asyncio.gather(
            self._warmup(),
            self._start_mcp(),
            return_exceptions=True,
        )

        self._app_context = build_app_context()
        self._emit({"type": "conversation", "status": "listening"})
        log.info("Conversation ready. Apps: %s", get_installed_apps())

    async def _warmup(self):
        t0 = time.time()
        try:
            silent = np.zeros(1600, dtype=np.float32)
            await asyncio.to_thread(self.transcriber.transcribe, silent)
        except Exception:
            pass
        log.info("Warmup done in %.1fs", time.time() - t0)

    async def _start_mcp(self):
        try:
            await self.mcp.start_servers()
        except Exception as e:
            log.warning("MCP: %s", e)

    async def process_audio(self, audio: np.ndarray):
        """Process a recorded audio chunk (called when user releases hotkey)."""
        if not self.active:
            return
        t0 = time.time()

        # Interrupt TTS if still playing
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

        if not transcript or len(transcript.strip()) < 2:
            self._emit({"type": "conversation", "status": "listening"})
            return

        log.info("User (%.0fms): %s", (time.time() - t0) * 1000, transcript)
        self._emit({"type": "conversation", "status": "thinking", "user_text": transcript})
        self.memory.add_to_session("user", transcript)

        # Route
        tier, instant_action = route(transcript)
        current_app = screen.get("active_app", "")

        if tier == "instant" and instant_action:
            result = await self.desktop.execute_instant(instant_action)
            narration = self._narrate_instant(instant_action, result)
            self._emit({"type": "conversation", "status": "speaking", "agent_text": narration})
            await self.tts.speak(narration)
            self.memory.add_to_session("assistant", narration)
            log.info("Tier 1 (%.0fms): %s", (time.time() - t0) * 1000, narration)

        elif tier == "groq" and self.groq:
            await self._call_groq(transcript, screen, mem_block, t0)

        else:
            if self.claude:
                await self._call_claude(transcript, screen, mem_block, t0)
            elif self.groq:
                await self._call_groq(transcript, screen, mem_block, t0)
            else:
                await self.tts.speak("No API key configured.")

        self._last_app = current_app
        self._emit({"type": "conversation", "status": "listening"})

        # Memory extraction
        self._turn_count += 1
        if self._turn_count % MEMORY_EXTRACT_INTERVAL == 0 and self.claude:
            asyncio.create_task(
                self.memory.extract_and_store(
                    self.memory.session_history[-MEMORY_EXTRACT_INTERVAL:], self.claude
                )
            )

    def _narrate_instant(self, action: dict, result: str) -> str:
        act = action.get("action", "")
        if act == "open_app":
            return f"Opening {action.get('app', 'app')}."
        elif act == "type":
            return "Typed."
        elif act == "undo":
            return "Undone."
        elif act == "screenshot":
            return "Screenshot taken."
        return "Done."

    async def _call_groq(self, transcript, screen, mem_block, t0):
        system = self._build_system(screen, mem_block, include_apps=True)
        messages = [{"role": "system", "content": system}]
        messages += self.memory.get_session_messages()[:-1]
        messages.append({"role": "user", "content": transcript})

        try:
            stream = self.groq.chat.completions.create(
                model=MODEL_FAST, messages=messages,
                max_tokens=512, temperature=0.7, stream=True,
            )

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
                                log.info("TTFS (%.0fms): %s", (time.time() - t0) * 1000, sentence[:50])
                                first = False
                            self._emit({"type": "conversation", "status": "speaking", "agent_text": sentence})
                            await self.tts.speak(sentence + ("." if sentence[-1] not in ".!?" else ""))
                            if not self.active:
                                break

            if buffer.strip() and self.active:
                await self.tts.speak(buffer.strip())

            if full.strip():
                self.memory.add_to_session("assistant", full.strip())

        except Exception as e:
            log.error("Groq: %s", e)
            await self.tts.speak("Sorry, I couldn't process that.")

    async def _call_claude(self, transcript, screen, mem_block, t0):
        system = self._build_system(screen, mem_block, include_apps=True)

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
                        result = await self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)[:2000],
                        })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            reply = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
            if reply:
                while self.tts.is_playing:
                    await asyncio.sleep(0.05)

                self.memory.add_to_session("assistant", reply)
                log.info("Agent (%.0fms): %s", (time.time() - t0) * 1000, reply[:80])
                self._emit({"type": "conversation", "status": "speaking", "agent_text": reply})

                sentences = SENTENCE_END.split(reply)
                for s in sentences:
                    s = s.strip()
                    if s and self.active:
                        await self.tts.speak(s + ("." if s[-1] not in ".!?" else ""))
            break

    async def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "run_applescript":
                return self.desktop.run_applescript(args["script"])
            elif name == "read_app_ui":
                return self.desktop.read_ui_tree(args["app_name"])
            elif name == "click_element":
                return self.desktop.click_element(args["app_name"], args["element_name"])
            elif name == "type_in_element":
                return self.desktop.type_in_element(args["app_name"], args["element_name"], args["text"])
            elif name == "type_text":
                return self.desktop.type_text(args["text"])
            elif name == "open_app":
                return self.desktop.open_app(args["app_name"])
            elif name == "press_shortcut":
                return self.desktop.press_shortcut(*args["keys"])
            elif name == "query_screen_memory":
                return self.desktop.query_screenpipe(args["query"], args.get("minutes_back", 60))
            elif name == "click_at_coordinates":
                return self.desktop.click(args["x"], args["y"])
            elif "__" in name:
                server, tool = name.split("__", 1)
                return await self.mcp.call_tool(server, tool, args)
            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {e}"

    def _build_system(self, screen, mem_block, include_apps=False) -> str:
        parts = [SYSTEM_PROMPT]
        if mem_block:
            parts.append(f"\nWHAT YOU REMEMBER:\n{mem_block}")
        if include_apps and self._app_context:
            parts.append(f"\n{self._app_context}")
        app = screen.get("active_app", "")
        win = screen.get("window_title", "")
        if app:
            parts.append(f"\nCURRENT: {app}" + (f" — {win}" if win else ""))
        return "\n".join(parts)

    async def stop(self):
        self.active = False
        self.tts.interrupt()
        await self.mcp.stop_servers()
        self._emit({"type": "conversation", "status": "stopped"})
        log.info("Conversation stopped")
