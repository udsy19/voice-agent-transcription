"""Tool definitions for Claude API.

These are passed to Claude so it can decide when to call them.
MCP tools are added dynamically at runtime.
"""

TOOLS = [
    {
        "name": "type_text",
        "description": "Type text at the current cursor position in any app",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to type"}},
            "required": ["text"],
        },
    },
    {
        "name": "open_app",
        "description": "Open or focus a macOS application",
        "input_schema": {
            "type": "object",
            "properties": {"app_name": {"type": "string", "description": "e.g. 'Slack', 'Safari', 'Calendar'"}},
            "required": ["app_name"],
        },
    },
    {
        "name": "press_shortcut",
        "description": "Press a keyboard shortcut",
        "input_schema": {
            "type": "object",
            "properties": {"keys": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['cmd', 'c'] for copy"}},
            "required": ["keys"],
        },
    },
    {
        "name": "run_applescript",
        "description": "Execute AppleScript for macOS automation: read emails, create events, interact with Finder, send messages, control any app.",
        "input_schema": {
            "type": "object",
            "properties": {"script": {"type": "string", "description": "AppleScript code to execute"}},
            "required": ["script"],
        },
    },
    {
        "name": "query_screen_memory",
        "description": "Search past screen activity and audio transcripts via Screenpipe. Use when user asks about something they saw or heard earlier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "minutes_back": {"type": "integer", "description": "How many minutes back to search", "default": 60},
            },
            "required": ["query"],
        },
    },
    {
        "name": "click_at",
        "description": "Click at specific screen coordinates. Use with screenshot context to identify what to click.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "description": {"type": "string", "description": "What you're clicking and why"},
            },
            "required": ["x", "y", "description"],
        },
    },
]
