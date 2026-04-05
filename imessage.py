"""iMessage integration via AppleScript + SQLite — read and send texts."""

import subprocess
import os
import sqlite3
from datetime import datetime
from logger import get_logger

log = get_logger("imessage")

# iMessage database path
IMESSAGE_DB = os.path.expanduser("~/Library/Messages/chat.db")
CONTACTS_DB = os.path.expanduser("~/Library/Application Support/AddressBook/Sources")


def _resolve_contact(phone_or_email: str) -> str:
    """Try to resolve a phone number/email to a contact name."""
    if not phone_or_email or phone_or_email == "Unknown":
        return "Unknown"

    # Try AddressBook database
    try:
        import glob
        db_files = glob.glob(os.path.join(CONTACTS_DB, "*/AddressBook-v22.abcddb"))
        for db_path in db_files:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            # Strip + and country code for matching
            clean_num = phone_or_email.replace("+1", "").replace("+", "").replace("-", "").replace(" ", "")[-10:]
            cursor.execute("""
                SELECT ZFIRSTNAME, ZLASTNAME FROM ZABCDRECORD
                WHERE ZFIRSTNAME IS NOT NULL
                AND ROWID IN (
                    SELECT ZOWNER FROM ZABCDPHONENUMBER WHERE REPLACE(REPLACE(REPLACE(REPLACE(ZFULLNUMBER,' ',''),'-',''),'(',''),')','') LIKE ?
                    UNION SELECT ZOWNER FROM ZABCDEMAILADDRESS WHERE ZADDRESS LIKE ?
                )
            """, (f"%{clean_num}%", f"%{phone_or_email}%"))
            row = cursor.fetchone()
            conn.close()
            if row:
                name = f"{row[0] or ''} {row[1] or ''}".strip()
                if name:
                    return name
    except Exception:
        pass

    # Fallback: return last 4 digits
    if phone_or_email.startswith("+") and len(phone_or_email) > 6:
        return f"...{phone_or_email[-4:]}"
    return phone_or_email


def _clean_message_text(text: str) -> str:
    """Clean message text for speech — remove emojis, URLs, special chars."""
    import re
    # Remove emojis
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0000FE00-\U0000FEFF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:200]


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
            sender = "me" if row[3] else _resolve_contact(row[0])
            messages.append({
                "from": sender,
                "text": _clean_message_text(row[1]),
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


def _find_phone_for_contact(name: str) -> list[str]:
    """Look up phone numbers for a contact name from AddressBook."""
    import glob
    phones = []
    try:
        ab_files = glob.glob(os.path.expanduser("~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb"))
        for ab in ab_files:
            conn = sqlite3.connect(f"file:{ab}?mode=ro", uri=True)
            cur = conn.cursor()
            # Case-insensitive + partial match (first 4 chars minimum)
            short = name[:4].lower() if len(name) > 3 else name.lower()
            cur.execute("""
                SELECT p.ZFULLNUMBER FROM ZABCDRECORD r
                JOIN ZABCDPHONENUMBER p ON p.ZOWNER = r.ROWID
                WHERE LOWER(r.ZFIRSTNAME) LIKE ? OR LOWER(r.ZLASTNAME) LIKE ?
                   OR LOWER(r.ZFIRSTNAME) LIKE ?
            """, (f"%{name.lower()}%", f"%{name.lower()}%", f"%{short}%"))
            for row in cur.fetchall():
                if row[0]:
                    # Normalize: strip spaces, parens, dashes
                    import re
                    clean = re.sub(r'[\s()\-]', '', row[0])
                    phones.append(clean)
            conn.close()
    except Exception:
        pass
    return phones


def get_messages_from(contact: str, count: int = 5) -> dict:
    """Get recent messages from a specific contact."""
    if not os.path.exists(IMESSAGE_DB):
        return {"ok": False, "error": "Messages database not found."}

    try:
        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Build search: contact name in chat display_name + phone numbers from AddressBook
        search_terms = [f"%{contact}%"]
        if len(contact) > 3:
            search_terms.append(f"%{contact[:4]}%")

        # Look up actual phone numbers from contacts
        phones = _find_phone_for_contact(contact)
        for p in phones:
            search_terms.append(f"%{p[-10:]}%")  # last 10 digits

        where_parts = []
        params = []
        for term in search_terms:
            where_parts.append("h.id LIKE ?")
            params.append(term)
            where_parts.append("c.display_name LIKE ?")
            params.append(term)
        params.append(count)

        cursor.execute(f"""
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
              AND ({' OR '.join(where_parts)})
            ORDER BY m.date DESC
            LIMIT ?
        """, params)

        messages = []
        for row in cursor.fetchall():
            sender = "me" if row[3] else _resolve_contact(row[0])
            messages.append({
                "from": sender,
                "text": _clean_message_text(row[1]),
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
