"""Conversation Agent — main loop.

VAD → Transcribe → Screen Context → Claude API (with tools) → TTS
"""

import asyncio
import os
import anthropic
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

You can:
- See their screen (screenshot + active app provided with every message)
- Hear and speak to them in real time
- Control their desktop (type, click, open apps, run AppleScript)
- Search their screen history via Screenpipe

Rules:
- Respond in natural spoken language. No markdown, no bullet points, no headers.
- Be concise. This is voice. 1-3 sentences unless more is needed.
- Use screen context to understand what the user is referring to without them explaining.
- Prefer taking action directly over explaining how to do it.
- Ask for confirmation before: sending emails, deleting things, making purchases.
- If you can see what they're talking about on screen, reference it directly.
- When you execute a tool, briefly say what you're doing."""

MODEL = os.getenv("CONVERSATION_MODEL", "claude-sonnet-4-20250514")
MEMORY_EXTRACT_INTERVAL = 10  # extract memories every N turns


class ConversationAgent:
    def __init__(self, emit_fn=None):
        """
        Args:
            emit_fn: callback to broadcast status to WebSocket clients
        """
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.vad = VADStream()
        self.context = ScreenContext()
        self.tts = StreamingTTS()
        self.memory = AgentMemory()
        self.mcp = MCPClient()
        self.desktop = DesktopController()
        self.transcriber = Transcriber(backend="groq")
        self.active = False
        self._emit = emit_fn or (lambda msg: None)
        self._turn_count = 0

    async def start(self):
        """Start the conversation loop."""
        self.active = True
        log.info("Conversation mode started")
        self._emit({"type": "conversation", "status": "starting"})

        # Start MCP servers in background
        try:
            await self.mcp.start_servers()
        except Exception as e:
            log.warning("MCP startup failed: %s", e)

        mcp_tools = await self.mcp.list_tools()
        all_tools = TOOLS + mcp_tools

        self._emit({"type": "conversation", "status": "listening"})

        try:
            async for audio_chunk in self.vad.start():
                if not self.active:
                    break

                # If TTS is playing and user speaks, interrupt
                if self.tts.is_playing:
                    self.tts.interrupt()
                    await asyncio.sleep(0.1)

                # Transcribe
                self._emit({"type": "conversation", "status": "transcribing"})
                transcript = self.transcriber.transcribe(audio_chunk)
                if not transcript or len(transcript.strip()) < 2:
                    self._emit({"type": "conversation", "status": "listening"})
                    continue

                log.info("User: %s", transcript)
                self._emit({"type": "conversation", "status": "thinking",
                             "user_text": transcript})

                # Capture screen context (fast, ~100ms)
                screen = self.context.capture()

                # Retrieve relevant long-term memories
                memory_block = self.memory.build_memory_block(transcript)

                # Build system prompt
                system = SYSTEM_PROMPT
                if memory_block:
                    system += f"\n\nWHAT YOU REMEMBER ABOUT THE USER:\n{memory_block}"
                if screen["active_app"]:
                    system += f"\n\nCURRENT CONTEXT:\n- Active app: {screen['active_app']}"
                if screen["window_title"]:
                    system += f"\n- Window: {screen['window_title']}"

                # Add to session
                self.memory.add_to_session("user", transcript)

                # Build user content with screenshot
                user_content = []
                if screen["screenshot_b64"]:
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screen["screenshot_b64"],
                        },
                    })
                user_content.append({"type": "text", "text": transcript})

                # Build messages
                messages = self.memory.get_session_messages()[:-1]  # exclude current
                messages.append({"role": "user", "content": user_content})

                # Call Claude
                try:
                    response = self.client.messages.create(
                        model=MODEL,
                        max_tokens=1024,
                        system=system,
                        messages=messages,
                        tools=all_tools if all_tools else anthropic.NOT_GIVEN,
                    )
                except Exception as e:
                    log.error("Claude API error: %s", e)
                    await self.tts.speak("Sorry, I had trouble processing that. Could you try again?")
                    self._emit({"type": "conversation", "status": "listening"})
                    continue

                # Handle tool calls
                while response.stop_reason == "tool_use":
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            log.info("Tool: %s(%s)", block.name, block.input)
                            result = await self._execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result),
                            })

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                    try:
                        response = self.client.messages.create(
                            model=MODEL,
                            max_tokens=1024,
                            system=system,
                            messages=messages,
                            tools=all_tools if all_tools else anthropic.NOT_GIVEN,
                        )
                    except Exception as e:
                        log.error("Claude API error during tool loop: %s", e)
                        break

                # Extract text response
                reply = " ".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                ).strip()

                if reply:
                    log.info("Agent: %s", reply)
                    self.memory.add_to_session("assistant", reply)
                    self._emit({"type": "conversation", "status": "speaking",
                                 "agent_text": reply})
                    await self.tts.speak(reply)

                self._emit({"type": "conversation", "status": "listening"})

                # Periodic memory extraction
                self._turn_count += 1
                if self._turn_count % MEMORY_EXTRACT_INTERVAL == 0:
                    recent = self.memory.session_history[-MEMORY_EXTRACT_INTERVAL:]
                    asyncio.create_task(
                        self.memory.extract_and_store(recent, self.client)
                    )

        except Exception as e:
            log.error("Conversation loop error: %s", e, exc_info=True)
        finally:
            await self.stop()

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool call from Claude."""
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
                return self.desktop.query_screenpipe(
                    args["query"], args.get("minutes_back", 60)
                )
            elif name == "click_at":
                return self.desktop.click(args["x"], args["y"])
            elif "__" in name:
                # MCP tool: server__tool_name
                server, tool = name.split("__", 1)
                return await self.mcp.call_tool(server, tool, args)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            log.error("Tool execution error: %s", e)
            return f"Error: {e}"

    async def stop(self):
        """Stop the conversation."""
        self.active = False
        self.tts.interrupt()
        await self.vad.stop()
        await self.mcp.stop_servers()
        self._emit({"type": "conversation", "status": "stopped"})
        log.info("Conversation mode stopped")
