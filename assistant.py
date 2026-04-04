"""Muse AI Assistant — voice-controlled tool-calling agent.

Handles commands like "add an event to my calendar" or "draft an email to John".
Uses Groq LLM with tool calling. Supports multi-turn conversations — if info
is missing (e.g., no date for a calendar event), it asks back via TTS and waits
for the user's next voice input to continue.
"""

import json
import time
import subprocess
from groq import Groq
from logger import get_logger

log = get_logger("assistant")

ASSISTANT_MODEL = "llama-3.3-70b-versatile"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event. Use when user wants to add, schedule, or create an event, meeting, or reminder. You MUST have a date/time before calling this — if the user didn't specify one, ask them first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start_time": {"type": "string", "description": "Start time in ISO 8601 (e.g., 2026-04-04T15:00:00)"},
                    "end_time": {"type": "string", "description": "End time in ISO 8601. Default 1 hour after start."},
                    "description": {"type": "string", "description": "Event description"},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "Attendee emails"},
                },
                "required": ["summary", "start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": "List calendar events. Returns events for today by default, or up to 14 days ahead if the user asks about 'next week', 'this week', 'upcoming', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "number", "description": "Number of days to look ahead. Use 1 for today, 7 for this week, 14 for next week."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Draft an email (does NOT send). Default to this unless user explicitly says 'send'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email immediately. ONLY when user explicitly says 'send'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_last_draft",
            "description": "Send the most recently created draft. Use when user says 'send it'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "List connected accounts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _speak(text: str):
    """Speak text aloud using Kokoro TTS (or macOS say fallback)."""
    try:
        import tts as tts_module
        import safe_json
        from config import DATA_DIR
        prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
        voice = prefs.get("voice", "af_heart")
        tts_module.speak(text, voice=voice)
    except Exception as e:
        log.warning("Kokoro TTS failed: %s, falling back to say", e)
        try:
            subprocess.Popen(
                ["say", "-v", "Samantha", "-r", "190", text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


class Assistant:
    """Voice-controlled AI assistant with tool calling and multi-turn memory."""

    def __init__(self, groq_client: Groq, oauth_manager, emit_fn):
        self._client = groq_client
        self._oauth = oauth_manager
        self._emit = emit_fn
        # Multi-turn conversation history (persists across recordings)
        self._conversation: list[dict] = []
        self._conversation_expires: float = 0

    def _prefetch_today(self) -> str:
        """Pre-fetch today's events so the LLM has context without a tool call."""
        try:
            token = self._oauth.get_token("google") if self._oauth else None
            if not token:
                return ""
            from integrations.google_calendar import list_events
            result = list_events(token, days_ahead=1, max_results=15)
            if result.get("ok") and result.get("events"):
                lines = []
                for e in result["events"]:
                    lines.append(f"- {e['summary']} ({e['start']} to {e['end']})")
                return "\n\nToday's calendar:\n" + "\n".join(lines)
        except Exception:
            pass
        return ""

    def handle(self, command: str) -> str | None:
        """Process a voice command. Supports multi-turn follow-ups."""
        if not self._client:
            return None

        # Expire stale conversations (10 min timeout)
        if time.time() > self._conversation_expires:
            self._conversation.clear()

        accounts = self._oauth.list_accounts() if self._oauth else []
        today_context = self._prefetch_today()
        system_prompt = self._build_system_prompt(accounts, today_context)

        # Build messages: system + conversation history + new user message
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._conversation)
        messages.append({"role": "user", "content": command})

        try:
            response = self._client.chat.completions.create(
                model=ASSISTANT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024,
                timeout=20,
            )

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls

            # No tool calls — LLM is asking a question or responding with text
            if not tool_calls:
                text = choice.message.content or ""
                if text:
                    # Save to conversation history
                    self._conversation.append({"role": "user", "content": command})
                    self._conversation.append({"role": "assistant", "content": text})
                    self._conversation_expires = time.time() + 600  # 2 min window

                    self._stream(text)
                    _speak(text)  # Speak the question/response aloud
                    return text
                return None

            # Execute tool calls
            self._conversation.append({"role": "user", "content": command})
            messages.append(choice.message)

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                self._emit({"type": "assistant_stream",
                            "text": f"Running {fn_name.replace('_', ' ')}...", "done": False})

                result = self._execute_tool(fn_name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

            # Generate final human-readable response
            final = self._client.chat.completions.create(
                model=ASSISTANT_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=512,
                timeout=20,
            )

            text = final.choices[0].message.content or "Done."

            # Save to conversation + stream + speak
            self._conversation.append({"role": "assistant", "content": text})
            self._conversation_expires = time.time() + 600

            # Keep conversation short (last 10 turns)
            if len(self._conversation) > 20:
                self._conversation = self._conversation[-20:]

            self._stream(text)
            _speak(text)
            return text

        except Exception as e:
            log.error("Assistant error: %s", e, exc_info=True)
            msg = f"Sorry, something went wrong."
            self._stream(msg)
            _speak(msg)
            return msg

    def _stream(self, text: str):
        """Stream text to the pill UI word-by-word."""
        words = text.split()
        buf = ""
        for i, word in enumerate(words):
            buf += word + " "
            if i % 3 == 2 or i == len(words) - 1:
                self._emit({
                    "type": "assistant_stream",
                    "text": buf.strip(),
                    "done": i == len(words) - 1,
                })
                time.sleep(0.03)

    def _execute_tool(self, name: str, args: dict) -> dict:
        """Execute a tool and return the result."""
        log.info("Tool: %s(%s)", name, json.dumps(args)[:200])

        if name == "list_accounts":
            accounts = self._oauth.list_accounts() if self._oauth else []
            if not accounts:
                return {"result": "No accounts connected. Go to Settings > Integrations."}
            return {"result": [f"{a['service']}: {a['email']}" for a in accounts]}

        token = self._oauth.get_token("google") if self._oauth else None
        if not token:
            return {"error": "No Google account connected. Go to Settings > Integrations."}

        if name == "create_calendar_event":
            from integrations.google_calendar import create_event
            return create_event(token, args["summary"], args["start_time"],
                                args.get("end_time", ""), args.get("description", ""),
                                args.get("attendees"))

        elif name == "list_calendar_events":
            from integrations.google_calendar import list_events
            try:
                days = max(1, min(30, int(float(args.get("days", 1)))))
            except (ValueError, TypeError):
                days = 1
            return list_events(token, days_ahead=days, max_results=25)

        elif name == "draft_email":
            from integrations.gmail import draft_email
            result = draft_email(token, args["to"], args["subject"], args["body"])
            if result.get("ok") and self._oauth:
                self._oauth._last_draft_id = result["draft_id"]
            return result

        elif name == "send_email":
            from integrations.gmail import send_email
            return send_email(token, args["to"], args["subject"], args["body"])

        elif name == "send_last_draft":
            if not self._oauth or not self._oauth._last_draft_id:
                return {"error": "No recent draft to send."}
            from integrations.gmail import send_draft
            result = send_draft(token, self._oauth._last_draft_id)
            if result.get("ok"):
                self._oauth._last_draft_id = None
            return result

        return {"error": f"Unknown tool: {name}"}

    def _build_system_prompt(self, accounts: list[dict], today_context: str = "") -> str:
        now = time.strftime("%A, %B %d, %Y at %I:%M %p")
        acct_list = "\n".join(f"- {a['service']}: {a['email']}" for a in accounts) if accounts else "None"

        return (
            f"You are a smart personal voice assistant. Right now: {now}\n"
            f"Connected accounts: {acct_list}\n"
            f"{today_context}\n"
            "\n## How to respond\n"
            "- Keep responses SHORT — 1-3 sentences max. This will be spoken aloud.\n"
            "- Be natural and conversational, like a real assistant.\n"
            "- When listing events, say them naturally: 'You have a flight to New York at 10:43am, then coffee with Teng at 1pm.'\n"
            "- Don't just list data — interpret it. 'Looks like a busy morning but your afternoon is free.'\n"
            "\n## Calendar intelligence\n"
            "- 'today' = days 1, 'this week' = days 7, 'next week' = days 14, 'this month' = days 30\n"
            "- 'tomorrow' means the next day. 'next Monday' means the coming Monday.\n"
            "- ALWAYS check for conflicts before creating. If there's an overlap, warn the user.\n"
            "- If they say 'am I free at 2?' — check the calendar and answer directly.\n"
            "- If they say 'move my 3pm to 4pm' — that requires delete + create (not yet supported, tell them).\n"
            "- If they say 'cancel the meeting' — ask which one if there are multiple.\n"
            "- If title is missing, ask. If time is missing, ask. If date is missing, ask.\n"
            "- When creating, confirm: 'Created Team Sync for tomorrow at 2pm.'\n"
            "\n## Email intelligence\n"
            "- Default to DRAFT unless user says 'send'. After drafting: 'Draft ready. Say send it.'\n"
            "- If no recipient, ask. If no subject, infer from body.\n"
            "- Write the email body professionally but match the user's tone.\n"
            "\n## Edge cases\n"
            "- If you don't understand, ask for clarification — never guess wrong.\n"
            "- If no accounts connected, say 'Connect Google in Settings first.'\n"
            "- If a tool fails, explain what went wrong simply.\n"
            "- If the user is just chatting (not calendar/email), respond briefly and naturally.\n"
        )
