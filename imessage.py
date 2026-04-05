"""iMessage integration via AppleScript + SQLite — read and send texts."""

import subprocess
import os
import sqlite3
from datetime import datetime
from logger import get_logger

log = get_logger("imessage")

# iMessage database path
IMESSAGE_DB = os.path.expanduser("~/Library/Messages/chat.db")


def get_recent_messages(count: int = 5) -> dict:
    """Get recent iMessages by reading the Messages database directly."""
    if not os.path.exists(IMESSAGE_DB):
        return {"ok": False, "error": "Messages database not found. Is Messages.app configured?"}

    try:
        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COALESCE(h.id, 'Unknown') as sender,
                m.text,
                datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as msg_date,
                m.is_from_me
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC
            LIMIT ?
        """, (count,))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                "from": "me" if row[3] else row[0],
                "text": row[1][:200],
                "time": row[2] or "",
            })

        conn.close()
        log.info("Got %d recent messages", len(messages))
        return {"ok": True, "messages": messages}

    except sqlite3.OperationalError as e:
        if "unable to open" in str(e).lower() or "permission" in str(e).lower():
            return {"ok": False, "error": "Can't access Messages database. Grant Full Disk Access to Terminal in System Settings > Privacy."}
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_messages_from(contact: str, count: int = 5) -> dict:
    """Get recent messages from a specific contact."""
    if not os.path.exists(IMESSAGE_DB):
        return {"ok": False, "error": "Messages database not found."}

    try:
        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COALESCE(h.id, 'Unknown') as sender,
                m.text,
                datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as msg_date,
                m.is_from_me
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
              AND (h.id LIKE ? OR c.display_name LIKE ?)
            ORDER BY m.date DESC
            LIMIT ?
        """, (f"%{contact}%", f"%{contact}%", count))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                "from": "me" if row[3] else row[0],
                "text": row[1][:200],
                "time": row[2] or "",
            })

        conn.close()
        return {"ok": True, "contact": contact, "messages": messages}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_message(to: str, text: str) -> dict:
    """Send an iMessage via AppleScript."""
    if not to or not text:
        return {"ok": False, "error": "Need both recipient and message."}

    import re
    to_clean = re.sub(r'[^a-zA-Z0-9@.+\- ]', '', to)
    text_clean = text.replace('"', '\\"').replace("'", "\\'")

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
            return {"ok": False, "error": result.stderr.strip()[:100] or "Failed to send. Is Messages.app open?"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
