"""Tool definitions for Claude API with speed/cost metadata.

Claude reads the descriptions to pick the fastest method.
Priority: AppleScript > Accessibility/Terminator > MCP > Vision click
"""

TOOLS = [
    {
        "name": "run_applescript",
        "description": (
            "Execute AppleScript for macOS automation. FASTEST method. "
            "USE THIS FIRST for: Mail, Calendar, Contacts, Finder, Safari, Messages, Notes, Reminders, Music, Terminal. "
            "Can read emails, create events, send messages, open URLs, control apps. "
            "Speed: 200-500ms. Always prefer over clicking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"script": {"type": "string", "description": "AppleScript code"}},
            "required": ["script"],
        },
    },
    {
        "name": "read_app_ui",
        "description": (
            "Read the entire UI tree of an open app. Returns all buttons, text fields, labels, menus as structured text. "
            "USE THIS to understand what's on screen before taking action. "
            "Speed: 80-200ms. Always call this before clicking in an unfamiliar app."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
    },
    {
        "name": "click_element",
        "description": (
            "Click a UI element by name in an app. "
            "Use element names from read_app_ui — do NOT guess coordinates. "
            "Speed: 80ms. Reliable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string"},
                "element_name": {"type": "string", "description": "Exact name from read_app_ui"},
            },
            "required": ["app_name", "element_name"],
        },
    },
    {
        "name": "type_in_element",
        "description": "Type text into a specific UI element (text field, search box). Speed: 80ms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string"},
                "element_name": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["app_name", "element_name", "text"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text at the current cursor position in any app. Speed: instant.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "open_app",
        "description": "Open or focus a macOS application by name. Speed: 200ms.",
        "input_schema": {
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
    },
    {
        "name": "press_shortcut",
        "description": "Press a keyboard shortcut. Speed: instant.",
        "input_schema": {
            "type": "object",
            "properties": {"keys": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['cmd','c']"}},
            "required": ["keys"],
        },
    },
    {
        "name": "query_screen_memory",
        "description": (
            "Search past screen activity and audio via Screenpipe. "
            "Use when user asks about something they saw/heard EARLIER. "
            "Do NOT use for current state — use read_app_ui instead. Speed: 100ms."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "minutes_back": {"type": "integer", "default": 60},
            },
            "required": ["query"],
        },
    },
    {
        "name": "click_at_coordinates",
        "description": (
            "Click at specific screen coordinates. LAST RESORT only. "
            "Use when NO other tool works (no AppleScript, no element name). "
            "Speed: 1-3s. Slow. Avoid if any other tool applies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "reason": {"type": "string", "description": "Why no faster method works"},
            },
            "required": ["x", "y", "reason"],
        },
    },
]

# Tool-specific spoken fillers while executing
TOOL_FILLERS = {
    "run_applescript": ["Checking...", "One sec...", "Looking that up..."],
    "read_app_ui": ["Reading the app...", "Scanning..."],
    "click_element": ["Clicking...", "On it..."],
    "type_in_element": ["Typing...", "On it..."],
    "query_screen_memory": ["Let me think back...", "Checking your history..."],
    "open_app": ["Opening...", "On it..."],
    "click_at_coordinates": ["Clicking...", "Navigating..."],
}
