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
            "name": "remember_fact",
            "description": "Save a fact about the user to long-term memory. ALWAYS use this (not add_todo) when user says 'remember', 'note that', 'keep in mind', or shares personal info about themselves, their preferences, relationships, or background.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The fact to remember about the user."}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event. MUST have title and start time. Ask user if missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title. Format nicely: capitalize, be descriptive."},
                    "start_time": {"type": "string", "description": "ISO 8601 start (e.g., 2026-04-04T15:00:00)"},
                    "end_time": {"type": "string", "description": "ISO 8601 end. Default 1hr after start."},
                    "description": {"type": "string", "description": "Notes/agenda for the event."},
                    "location": {"type": "string", "description": "Location or address."},
                    "attendees": {"type": "string", "description": "Comma-separated email addresses of invitees."},
                    "timezone": {"type": "string", "description": "IANA timezone. Default America/New_York."},
                    "add_meet": {"type": "string", "description": "'true' to add a Google Meet link. Use when user says 'add a meet link' or 'video call'."},
                },
                "required": ["summary", "start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": "List calendar events. today=1 day, this week=7, next week=14, this month=30.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "string", "description": "Days ahead to look. Default '1'."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": "Update an existing event (rename, reschedule, add attendees, change location, add notes). Need the event_id from list_calendar_events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The event ID to update."},
                    "summary": {"type": "string", "description": "New title."},
                    "start_time": {"type": "string", "description": "New start time ISO 8601."},
                    "end_time": {"type": "string", "description": "New end time ISO 8601."},
                    "description": {"type": "string", "description": "New notes/description."},
                    "location": {"type": "string", "description": "New location."},
                    "attendees": {"type": "string", "description": "Comma-separated emails to set as attendees."},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete/cancel a calendar event. Need event_id from list_calendar_events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The event ID to delete."},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Draft an email (does NOT send). Default to this unless user says 'send'.",
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
            "description": "Send email immediately. ONLY when user explicitly says 'send'.",
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
            "description": "Send the most recently created draft.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Add a task to the todo list. Use for reminders, tasks, things to do.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The task description."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_todo",
            "description": "Mark a task as done/completed. Use when user says they finished something, 'check off X', 'done with X', 'bought X'.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The task text to mark as done (fuzzy match)."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_todos",
            "description": "List pending tasks/todos.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_meeting_notes",
            "description": "Save meeting notes with action items. Use when user says 'meeting notes' or describes what happened in a meeting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Meeting title or who it was with."},
                    "notes": {"type": "string", "description": "Key points discussed."},
                    "action_items": {"type": "string", "description": "Comma-separated action items."},
                },
                "required": ["summary", "notes"],
            },
        },
    },
]


def _speak(text: str):
    """Speak text via Kokoro TTS."""
    try:
        import tts as tts_module
        import safe_json
        from config import DATA_DIR
        prefs = safe_json.load(str(DATA_DIR / "preferences.json"), {})
        voice = prefs.get("voice", "af_heart")
        tts_module.speak(text, voice=voice)
    except Exception as e:
        log.warning("TTS failed: %s", e)


class Assistant:
    """Voice-controlled AI assistant with tool calling and multi-turn memory."""

    def __init__(self, groq_client: Groq, oauth_manager, emit_fn, todos=None, brain=None):
        self._client = groq_client
        self._oauth = oauth_manager
        self._emit = emit_fn
        self._todos = todos
        self._brain = brain
        # Multi-turn conversation history (persists across recordings)
        self._conversation: list[dict] = []
        self._conversation_expires: float = 0

    _cached_today: str = ""
    _cached_today_ts: float = 0

    def _prefetch_today(self) -> str:
        """Get today's events — cached for 60s to avoid hammering Google API."""
        if time.time() - self._cached_today_ts < 60 and self._cached_today:
            return self._cached_today
        try:
            token = self._oauth.get_token("google") if self._oauth else None
            if not token:
                return self._cached_today
            from integrations.google_calendar import list_events
            result = list_events(token, days_ahead=1, max_results=15)
            if result.get("ok") and result.get("events"):
                lines = [f"- {e['summary']} ({e['start']} to {e['end']})" for e in result["events"]]
                self._cached_today = "\n\nToday's calendar:\n" + "\n".join(lines)
            else:
                self._cached_today = ""
            self._cached_today_ts = time.time()
        except Exception:
            pass  # use stale cache on network error
        return self._cached_today

    def handle(self, command: str) -> str | None:
        """Process a voice command. Supports multi-turn follow-ups."""
        if not self._client:
            return None

        if time.time() > self._conversation_expires:
            self._conversation.clear()

        accounts = self._oauth.list_accounts() if self._oauth else []
        today_context = self._prefetch_today()
        brain_context = self._brain.get_context_for_llm() if self._brain else ""
        todos_context = self._todos.summary_for_llm() if self._todos else ""
        # Recall relevant memories for this query
        import memory as mem
        mem_context = mem.get_context_for_llm(command)
        system_prompt = self._build_system_prompt(
            accounts, today_context + brain_context + todos_context + mem_context
        )

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
                temperature=0.2,
                max_tokens=512,
                timeout=15,
            )

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls

            # No tool calls — LLM is asking a question or responding with text
            if not tool_calls:
                text = choice.message.content or ""
                if text:
                    self._conversation.append({"role": "user", "content": command})
                    self._conversation.append({"role": "assistant", "content": text})
                    self._conversation_expires = time.time() + 600
                    # Only save to memory if it contains personal/preference info
                    if any(w in command.lower() for w in ['prefer','like','love','hate','always','never','name is','my ','remember','favorite','i am','i\'m']):
                        mem.remember_async(command)

                    self._stream(text)
                    import threading
                    threading.Thread(target=_speak, args=(text,), daemon=True).start()
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

                # If this was a calendar list, emit events for pill calendar view
                if fn_name == "list_calendar_events" and isinstance(result, dict) and result.get("ok"):
                    events = result.get("events", [])
                    if events:
                        self._emit({"type": "calendar_view", "events": events[:8]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

            # Fast model for summarizing tool results into speech
            final = self._client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.3,
                max_tokens=200,
                timeout=10,
            )

            text = final.choices[0].message.content or "Done."

            self._conversation.append({"role": "assistant", "content": text})
            self._conversation_expires = time.time() + 600
            if len(self._conversation) > 20:
                self._conversation = self._conversation[-20:]

            # Only save to memory if user shared personal info
            if any(w in command.lower() for w in ['prefer','like','love','hate','always','never','name is','my ','remember','favorite','i am','i\'m']):
                mem.remember_async(command)

            self._stream(text)
            import threading
            threading.Thread(target=_speak, args=(text,), daemon=True).start()
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

        token = self._oauth.get_token("google") if self._oauth else None
        if not token:
            return {"error": "No Google account connected. Go to Settings > Integrations."}

        if name == "create_calendar_event":
            from integrations.google_calendar import create_event
            att_str = args.get("attendees", "")
            attendees = [e.strip() for e in att_str.split(",") if e.strip()] if att_str else None
            return create_event(
                token, args["summary"], args["start_time"],
                end_time=args.get("end_time", ""),
                description=args.get("description", ""),
                location=args.get("location", ""),
                attendees=attendees,
                timezone=args.get("timezone", "America/New_York"),
                add_meet=str(args.get("add_meet", "")).lower() == "true",
            )

        elif name == "list_calendar_events":
            from integrations.google_calendar import list_events
            try:
                days = max(1, min(30, int(float(args.get("days", 1)))))
            except (ValueError, TypeError):
                days = 1
            return list_events(token, days_ahead=days, max_results=25)

        elif name == "update_calendar_event":
            from integrations.google_calendar import update_event
            fields = {}
            for k in ["summary", "description", "location", "start_time", "end_time"]:
                if args.get(k):
                    fields[k] = args[k]
            if args.get("attendees"):
                fields["attendees"] = [e.strip() for e in args["attendees"].split(",") if e.strip()]
            return update_event(token, args["event_id"], **fields)

        elif name == "delete_calendar_event":
            from integrations.google_calendar import delete_event
            return delete_event(token, args["event_id"])

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

        elif name == "remember_fact":
            import memory as mem_module
            m = mem_module._get_mem0()
            if m:
                try:
                    result = m.add(args["fact"], user_id="user")
                    entries = result.get("results", [])
                    log.info("Remembered via tool: %s (%d facts)", args["fact"][:50], len(entries))
                    return {"ok": True, "fact": args["fact"], "stored": len(entries)}
                except Exception as e:
                    log.error("Remember failed: %s", e)
                    return {"ok": False, "error": str(e)}
            return {"ok": False, "error": "Memory not available"}

        elif name == "add_todo":
            if self._todos:
                item = self._todos.add(args["text"])
                self._emit({"type": "todo_added", "item": item})
                return {"ok": True, "text": args["text"]}
            return {"error": "Todos not available."}

        elif name == "complete_todo":
            if self._todos:
                search = args["text"].lower()
                for item in self._todos.list_pending():
                    if search in item["text"].lower() or item["text"].lower() in search:
                        self._todos.complete(item["id"])
                        self._emit({"type": "todo_completed", "id": item["id"]})
                        return {"ok": True, "completed": item["text"]}
                return {"ok": False, "error": f"No pending task matching '{args['text']}'"}
            return {"error": "Todos not available."}

        elif name == "list_todos":
            if self._todos:
                pending = self._todos.list_pending()
                if not pending:
                    return {"result": "No pending tasks."}
                return {"result": [t["text"] for t in pending]}
            return {"result": "Todos not available."}

        elif name == "save_meeting_notes":
            if self._brain:
                actions = [a.strip() for a in args.get("action_items", "").split(",") if a.strip()]
                self._brain.add_meeting_notes(
                    args["summary"], time.strftime("%Y-%m-%d"),
                    args["notes"], actions,
                )
                # Auto-create todos from action items
                if actions and self._todos:
                    for a in actions:
                        item = self._todos.add(a)
                        self._emit({"type": "todo_added", "item": item})
                return {"ok": True, "summary": args["summary"], "action_items": actions}
            return {"error": "Brain not available."}

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
            "- When listing events, say times naturally: '10:43 AM' not '10:43:00', '12 PM' not '12:00:00', '1 PM' not '13:00'.\n"
            "- Say events naturally: 'You have a flight to New York at 10:43 AM, then coffee with Teng at 1 PM.'\n"
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
            "- When user says 'remember X' or shares personal info, use remember_fact (NOT add_todo).\n"
            "  add_todo is ONLY for action items like 'buy groceries' or 'call dentist'.\n"
            "- USE your memories proactively — if you know who someone is, use that knowledge.\n"
            "  Example: if user says 'email Samyukta' and you remember her email, use it.\n"
            "  Example: if user has a meeting with someone you know, mention relevant context.\n"
            "- When summarizing a conversation, extract key facts for memory.\n"
        )
