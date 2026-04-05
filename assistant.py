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
            "name": "forget_fact",
            "description": "Delete/forget a memory. Use when user says 'forget X', 'that's wrong', 'remove X from memory', or corrects a previous fact.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to forget — search term to find and delete the memory."}},
                "required": ["query"],
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
    # ── Power features ──
    {
        "type": "function",
        "function": {
            "name": "transform_clipboard",
            "description": "Read selected/clipboard text and transform it. Use for 'explain this', 'summarize', 'translate to X', 'make this professional', 'rewrite this'.",
            "parameters": {
                "type": "object",
                "properties": {"instruction": {"type": "string", "description": "How to transform the text."}},
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open a macOS application. 'Open Slack', 'Launch Safari', etc.",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string", "description": "App name: Slack, Safari, Notes, Finder, etc."}},
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_command",
            "description": "macOS system control: set volume, toggle DND, quit an app, run a Shortcut.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "One of: set_volume, toggle_dnd, quit_app, run_shortcut"},
                    "value": {"type": "string", "description": "Volume 0-100, app name, or shortcut name."},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Search past dictations and conversations. 'What did I say about X?', 'Find when I mentioned Y'.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weekly_reflection",
            "description": "Summarize this week: meetings, completed tasks, learnings, deadlines. 'How was my week?'",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_screen",
            "description": "Take a screenshot and analyze what's on screen. 'Summarize this page', 'What am I looking at?', 'Read this error'.",
            "parameters": {
                "type": "object",
                "properties": {"instruction": {"type": "string", "description": "What to do with the screenshot."}},
                "required": ["instruction"],
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
        if voice not in ("af_heart","af_alloy","af_bella","af_nicole","af_nova","af_sky","am_adam","am_eric","am_michael","bf_emma","bm_daniel","bm_george"):
            voice = "af_heart"
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

        # Only fetch expensive context when likely needed
        cmd_lower = command.lower()
        extra_context = ""

        # Calendar context — only if query mentions time/schedule/calendar/meeting
        if any(w in cmd_lower for w in ['calendar','schedule','today','tomorrow','week','meeting','event','free','busy','flight']):
            extra_context += self._prefetch_today()

        # Todos context — only if query mentions tasks/todos/list
        if any(w in cmd_lower for w in ['todo','task','list','remind','grocery','shopping','buy']):
            extra_context += self._todos.summary_for_llm() if self._todos else ""

        # Memory context — only for questions about people/preferences/facts (NOT screen analysis)
        if any(w in cmd_lower for w in ['who','what','my','favorite','prefer','remember','know','friend','girlfriend']) \
           and not any(w in cmd_lower for w in ['screen','looking at','page','see','summarize this']):
            import memory as mem
            extra_context += mem.get_context_for_llm(command)

        system_prompt = self._build_system_prompt(accounts, extra_context)

        # Build messages: system + conversation history + new user message
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._conversation)
        messages.append({"role": "user", "content": command})

        try:
            # Try 70b first, fall back to 8b on rate limit, manual exec on tool format error
            response = None
            for model in [ASSISTANT_MODEL, "llama-3.1-8b-instant"]:
                try:
                    response = self._client.chat.completions.create(
                        model=model, messages=messages, tools=TOOLS,
                        tool_choice="auto", temperature=0.2, max_tokens=512, timeout=15,
                    )
                    break  # success
                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate_limit" in err:
                        log.warning("%s rate limited, trying next model", model)
                        continue
                    elif "tool_use_failed" in err or "400" in err:
                        # Parse failed tool call and execute manually
                        log.warning("Tool format error, executing manually")
                        import re as _re
                        fn_match = _re.search(r'<function=(\w+)', err)
                        if fn_match:
                            json_match = _re.search(r'\{[^}]+\}', err)
                            try:
                                args = json.loads(json_match.group(0)) if json_match else {}
                            except Exception:
                                args = {}
                            result = self._execute_tool(fn_match.group(1), args)
                            messages.append({"role": "user", "content": f"Result: {json.dumps(result)[:200]}. Summarize in 1 sentence."})
                            response = self._client.chat.completions.create(
                                model="llama-3.1-8b-instant", messages=messages,
                                temperature=0.3, max_tokens=100, timeout=10,
                            )
                        else:
                            response = self._client.chat.completions.create(
                                model="llama-3.1-8b-instant", messages=messages,
                                temperature=0.3, max_tokens=256, timeout=10,
                            )
                        break
                    else:
                        raise

            if not response:
                self._stream("I'm being rate limited right now. Try again in a minute.")
                _speak("I'm being rate limited. Try again in a minute.")
                return "Rate limited. Try again shortly."

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls

            # No tool calls — LLM is asking a question or responding with text
            if not tool_calls:
                text = choice.message.content or ""
                if text:
                    self._conversation.append({"role": "user", "content": command})
                    self._conversation.append({"role": "assistant", "content": text})
                    self._conversation_expires = time.time() + 600
                    # Don't auto-save — memory is saved explicitly via remember_fact tool

                    self._stream(text)
                    import threading
                    threading.Thread(target=_speak, args=(text,), daemon=True).start()
                    return text
                return None

            # Execute tool calls with agentic retry
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

                # Agentic execution: try, and if it fails, adapt
                result = self._execute_tool(fn_name, args)

                if isinstance(result, dict) and result.get("error"):
                    error_msg = result["error"]
                    log.warning("Tool %s failed: %s — trying to recover", fn_name, error_msg[:60])

                    # Strategy 1: If it's a missing arg, ask the LLM to fix it
                    if "missing" in error_msg.lower() or "required" in error_msg.lower():
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": json.dumps(result)})
                        recovery = self._client.chat.completions.create(
                            model="llama-3.1-8b-instant", messages=messages,
                            tools=TOOLS, tool_choice="auto",
                            temperature=0.2, max_tokens=256, timeout=10,
                        )
                        if recovery.choices[0].message.tool_calls:
                            rtc = recovery.choices[0].message.tool_calls[0]
                            try:
                                rargs = json.loads(rtc.function.arguments)
                            except Exception:
                                rargs = {}
                            result = self._execute_tool(rtc.function.name, rargs)
                            log.info("Recovery succeeded: %s", rtc.function.name)
                        continue

                    # Strategy 2: If it's an API/network error, retry once after 2s
                    if any(w in error_msg.lower() for w in ["timeout", "connection", "429", "rate"]):
                        self._emit({"type": "assistant_stream",
                                    "text": "Retrying...", "done": False})
                        time.sleep(2)
                        result = self._execute_tool(fn_name, args)
                        if isinstance(result, dict) and result.get("error"):
                            log.warning("Retry also failed: %s", result["error"][:60])

                    # Strategy 3: If Google token expired, refresh and retry
                    if "401" in error_msg or "expired" in error_msg.lower():
                        if self._oauth:
                            self._oauth._token_cache.clear()
                            result = self._execute_tool(fn_name, args)

                # Calendar view
                if fn_name == "list_calendar_events" and isinstance(result, dict) and result.get("ok"):
                    events = result.get("events", [])
                    if events:
                        self._emit({"type": "calendar_view", "events": events[:8]})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)[:2000],
                })

            # Fast model for summarizing tool results into speech
            try:
                final = self._client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=200,
                    timeout=10,
                )
                text = final.choices[0].message.content or "Done."
            except Exception as sum_err:
                log.warning("Summary failed: %s — using raw result", sum_err)
                # Fallback: use the last tool result as text
                last_result = messages[-1].get("content", "Done.") if messages else "Done."
                try:
                    r = json.loads(last_result)
                    text = r.get("summary", r.get("analysis", r.get("fact", str(r.get("ok", "Done.")))))
                except Exception:
                    text = "Done."

            self._conversation.append({"role": "assistant", "content": text})
            self._conversation_expires = time.time() + 600
            if len(self._conversation) > 20:
                self._conversation = self._conversation[-20:]

            # Don't auto-save — memory is saved explicitly via remember_fact tool

            self._stream(text)
            import threading
            threading.Thread(target=_speak, args=(text,), daemon=True).start()
            return text

        except Exception as e:
            log.error("Assistant error: %s", e, exc_info=True)
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                msg = "I'm being rate limited right now. Give me a few seconds and try again."
            elif "timeout" in err or "timed out" in err:
                msg = "That took too long. Let me try a simpler approach — could you rephrase?"
            elif "connection" in err or "network" in err:
                msg = "I can't reach the server right now. Check your internet connection."
            elif "not found" in err or "not available" in err:
                msg = "That feature isn't available right now. Try something else."
            elif "tool_use_failed" in err:
                msg = "I had trouble executing that. Let me try differently — could you rephrase?"
            else:
                msg = f"Something went wrong. Try again in a moment."
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

        # Non-Google tools first (don't need OAuth token)
        if name == "remember_fact":
            import memory as mem_module
            fact_text = args["fact"]
            existing = mem_module.recall(fact_text, limit=3)
            for ex in existing:
                if ex.get("score", 0) > 0.5:
                    mem_module.delete(ex["id"])
                    log.info("Replacing similar memory: %s", ex.get("memory", "")[:40])
            m = mem_module._get_mem0()
            if m:
                try:
                    import uuid, hashlib
                    embedding = m.embedding_model.embed(fact_text)
                    m.vector_store.insert(
                        vectors=[embedding],
                        payloads=[{"data": fact_text, "hash": hashlib.md5(fact_text.encode()).hexdigest(),
                                   "user_id": "user", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00")}],
                        ids=[str(uuid.uuid4())],
                    )
                    log.info("Remembered: %s", fact_text[:50])
                    self._emit({"type": "memory_updated"})
                    return {"ok": True, "fact": fact_text}
                except Exception as e:
                    log.error("Remember failed: %s", e)
                    return {"ok": False, "error": str(e)}
            return {"ok": False, "error": "Memory not available"}

        elif name == "forget_fact":
            import memory as mem_module
            results = mem_module.recall(args["query"], limit=3)
            if results:
                deleted = []
                for r in results:
                    if r.get("id"):
                        mem_module.delete(r["id"])
                        deleted.append(r.get("memory", ""))
                self._emit({"type": "memory_updated"})
                return {"ok": True, "deleted": deleted}
            return {"ok": False, "error": "No matching memory found."}

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
                return {"result": [t["text"] for t in pending]} if pending else {"result": "No pending tasks."}
            return {"result": "Todos not available."}

        elif name == "save_meeting_notes":
            if self._brain:
                actions = [a.strip() for a in args.get("action_items", "").split(",") if a.strip()]
                self._brain.add_meeting_notes(args["summary"], time.strftime("%Y-%m-%d"), args["notes"], actions)
                if actions and self._todos:
                    for a in actions:
                        item = self._todos.add(a)
                        self._emit({"type": "todo_added", "item": item})
                return {"ok": True, "summary": args["summary"], "action_items": actions}
            return {"error": "Brain not available."}

        # Google tools — need OAuth token
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
            result = send_email(token, args["to"], args["subject"], args["body"])
            if result.get("ok"):
                import follow_ups
                follow_ups.add(args["to"], args["subject"], result.get("message_id", ""))
            return result

        elif name == "send_last_draft":
            if not self._oauth or not self._oauth._last_draft_id:
                return {"error": "No recent draft to send."}
            from integrations.gmail import send_draft
            result = send_draft(token, self._oauth._last_draft_id)
            if result.get("ok"):
                self._oauth._last_draft_id = None
            return result

        # ── Power features (no Google token needed) ──

        elif name == "transform_clipboard":
            from injector import get_selected_text, inject_text
            from cleaner import Cleaner
            selected = get_selected_text()
            if not selected:
                # Try clipboard
                import subprocess
                try:
                    r = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2)
                    selected = r.stdout.strip()
                except Exception:
                    pass
            if not selected:
                return {"error": "No text selected or in clipboard."}
            client = self._client
            if client:
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Transform the text as instructed. Return ONLY the result."},
                        {"role": "user", "content": f"Text:\n{selected[:3000]}\n\nInstruction: {args['instruction']}"},
                    ],
                    temperature=0.3, max_tokens=2048, timeout=15,
                )
                result = resp.choices[0].message.content.strip()
                inject_text(result)
                return {"ok": True, "chars": len(result)}
            return {"error": "No LLM available."}

        elif name == "open_app":
            import system_control
            return system_control.open_app(args["app_name"])

        elif name == "system_command":
            import system_control
            action = args.get("action", "")
            value = args.get("value", "")
            if action == "set_volume":
                return system_control.set_volume(int(value) if value else 50)
            elif action == "toggle_dnd":
                return system_control.toggle_dnd()
            elif action == "quit_app":
                return system_control.quit_app(value)
            elif action == "run_shortcut":
                return system_control.run_shortcut(value)
            return {"error": f"Unknown action: {action}"}

        elif name == "search_history":
            import memory as mem_module
            results = mem_module.recall(args["query"], limit=5)
            if results:
                return {"ok": True, "results": [{"text": r["memory"], "score": round(r.get("score", 0), 2)} for r in results]}
            return {"ok": True, "results": [], "message": "Nothing found matching that query."}

        elif name == "weekly_reflection":
            import reflection
            summary = reflection.compose_week(
                brain=self._brain, todos=self._todos, oauth=self._oauth,
            )
            return {"ok": True, "summary": summary}

        elif name == "analyze_screen":
            import vision
            return vision.analyze_screen(args["instruction"])

        return {"error": f"Unknown tool: {name}"}

    def _build_system_prompt(self, accounts: list[dict], today_context: str = "") -> str:
        now = time.strftime("%A, %B %d, %Y at %I:%M %p")
        acct_list = ", ".join(f"{a['service']}:{a['email']}" for a in accounts) if accounts else "None"

        return (
            f"You are a smart personal voice assistant. Right now: {now}\n"
            f"Connected accounts: {acct_list}\n"
            f"{today_context}\n"
            "\n## How to respond\n"
            "- Your response will be SPOKEN ALOUD by a voice synthesizer.\n"
            "- Keep responses SHORT — 1-2 sentences max.\n"
            "- Write how you'd SPEAK, not how you'd type:\n"
            "  GOOD: 'You have a meeting with Teng at 1 PM, then you're free.'\n"
            "  BAD: 'You have the following events: 1. Meeting with Teng Zhang - RGTI at 13:00:00'\n"
            "- Use natural time: '1 PM' not '13:00', '10:45 AM' not '10:45:00'\n"
            "- Use natural dates: 'next Thursday' not 'April 10th, 2026'\n"
            "- No bullet points, numbered lists, or markdown — flowing sentences only.\n"
            "- No event IDs, calendar IDs, or technical details.\n"
            "- Be warm and conversational, like a real human assistant.\n"
            "\n## Calendar\n"
            "- 'today' = days 1, 'this week' = days 7, 'next week' = days 14, 'this month' = days 30\n"
            "- Check for conflicts before creating. If overlap, warn the user.\n"
            "- If title or time is missing, ask. When creating, confirm naturally.\n"
            "\n## Email\n"
            "- Default to DRAFT unless user says 'send'. After drafting: 'Draft ready. Say send it.'\n"
            "- If no recipient, ask. Infer subject from body if not given.\n"
            "\n## Memory\n"
            "- remember_fact: for personal info, preferences, relationships, facts about the user.\n"
            "- add_todo: ONLY for action items like 'buy groceries' or 'call dentist'.\n"
            "- forget_fact: to delete wrong/outdated memories.\n"
            "- Corrections ('not X, it's Y'): call forget_fact THEN remember_fact together.\n"
            "- Same person with new info auto-replaces the old memory.\n"
            "- Use memories proactively — if you know someone's email, use it.\n"
            "\n## General\n"
            "- If you don't understand, ask for clarification.\n"
            "- If no accounts connected, say 'Connect Google in Settings.'\n"
            "- If a tool fails, explain simply.\n"
            "- For casual chat, respond briefly and warmly.\n"
        )
