"""iMessage integration via AppleScript — read and send texts."""

import subprocess
from logger import get_logger

log = get_logger("imessage")


def get_recent_messages(count: int = 5) -> dict:
    """Get the most recent iMessages."""
    script = f'''
        tell application "Messages"
            set msgList to {{}}
            set chatList to chats
            repeat with c in chatList
                if (count of messages of c) > 0 then
                    set lastMsg to last message of c
                    set senderName to name of c
                    set msgText to text of lastMsg
                    set msgDate to date of lastMsg
                    set end of msgList to senderName & " | " & msgText & " | " & (msgDate as string)
                end if
                if (count of msgList) >= {count} then exit repeat
            end repeat
            return msgList
        end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip() or "Messages app may not be running"}

        raw = result.stdout.strip()
        if not raw:
            return {"ok": True, "messages": []}

        messages = []
        for line in raw.split(", "):
            parts = line.split(" | ", 2)
            if len(parts) >= 2:
                messages.append({
                    "from": parts[0].strip(),
                    "text": parts[1].strip(),
                    "time": parts[2].strip() if len(parts) > 2 else "",
                })
        log.info("Got %d recent messages", len(messages))
        return {"ok": True, "messages": messages}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Messages app took too long to respond"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_message(to: str, text: str) -> dict:
    """Send an iMessage to a phone number or email."""
    if not to or not text:
        return {"ok": False, "error": "Need both recipient and message text."}

    # Sanitize to prevent AppleScript injection
    import re
    to_clean = re.sub(r'[^a-zA-Z0-9@.+\- ]', '', to)
    text_clean = text.replace('"', '\\"').replace('\n', '\\n')

    script = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{to_clean}" of targetService
            send "{text_clean}" to targetBuddy
        end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            log.info("Sent iMessage to %s", to_clean)
            return {"ok": True, "to": to_clean}
        else:
            return {"ok": False, "error": result.stderr.strip() or "Failed to send"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_messages_from(contact: str, count: int = 5) -> dict:
    """Get recent messages from a specific contact."""
    script = f'''
        tell application "Messages"
            set msgList to {{}}
            set chatList to chats
            repeat with c in chatList
                if name of c contains "{contact}" then
                    set msgs to messages of c
                    set msgCount to count of msgs
                    set startIdx to msgCount - {count} + 1
                    if startIdx < 1 then set startIdx to 1
                    repeat with i from startIdx to msgCount
                        set m to item i of msgs
                        set msgText to text of m
                        set msgDate to date of m
                        set sender to "them"
                        if (is_from_me of m) then set sender to "me"
                        set end of msgList to sender & " | " & msgText & " | " & (msgDate as string)
                    end repeat
                    exit repeat
                end if
            end repeat
            return msgList
        end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip()}

        raw = result.stdout.strip()
        messages = []
        for line in raw.split(", "):
            parts = line.split(" | ", 2)
            if len(parts) >= 2:
                messages.append({
                    "from": parts[0].strip(),
                    "text": parts[1].strip(),
                    "time": parts[2].strip() if len(parts) > 2 else "",
                })
        return {"ok": True, "contact": contact, "messages": messages}
    except Exception as e:
        return {"ok": False, "error": str(e)}
