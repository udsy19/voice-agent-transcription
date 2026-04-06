"""iMessage integration — read from chat.db, send via Shortcuts, contacts via native framework.

Architecture:
- READ: SQLite from ~/Library/Messages/chat.db (fast, reliable)
- SEND: macOS Shortcuts app (more reliable than AppleScript)
- CONTACTS: Native Contacts framework via pyobjc (proper API, not SQLite hacking)
- REAL-TIME: Poll chat.db mtime for new messages
"""

import os
import re
import sqlite3
import subprocess
import time
import threading
from logger import get_logger

log = get_logger("imessage")

IMESSAGE_DB = os.path.expanduser("~/Library/Messages/chat.db")
_contact_cache: dict[str, str] = {}  # phone → name, cached for session
_contact_cache_ts: float = 0


# ── Contact Resolution (Native Contacts Framework) ─────────────────────────

def _load_contacts():
    """Load contacts via native macOS Contacts framework. Cached for 5 min."""
    global _contact_cache, _contact_cache_ts
    if time.time() - _contact_cache_ts < 300 and _contact_cache:
        return

    try:
        import Contacts
        store = Contacts.CNContactStore.alloc().init()
        keys = [Contacts.CNContactGivenNameKey, Contacts.CNContactFamilyNameKey,
                Contacts.CNContactPhoneNumbersKey, Contacts.CNContactEmailAddressesKey]

        # Fetch all contacts
        request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)
        results = []

        def handler(contact, stop):
            name = f"{contact.givenName()} {contact.familyName()}".strip()
            if not name:
                return
            for phone in contact.phoneNumbers():
                num = phone.value().stringValue()
                clean = re.sub(r'[\s()\-+]', '', num)
                if len(clean) >= 10:
                    _contact_cache[clean[-10:]] = name  # last 10 digits as key
            for email in contact.emailAddresses():
                _contact_cache[email.value()] = name

        store.enumerateContactsWithFetchRequest_error_usingBlock_(request, None, handler)
        _contact_cache_ts = time.time()
        log.info("Loaded %d contact mappings", len(_contact_cache))

    except ImportError:
        log.warning("Contacts framework not available — install pyobjc-framework-Contacts")
        _load_contacts_fallback()
    except Exception as e:
        log.warning("Contacts load failed: %s — using fallback", e)
        _load_contacts_fallback()


def _load_contacts_fallback():
    """Fallback: load contacts from AddressBook SQLite."""
    global _contact_cache, _contact_cache_ts
    import glob
    try:
        ab_files = glob.glob(os.path.expanduser(
            "~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb"))
        for ab in ab_files:
            conn = sqlite3.connect(f"file:{ab}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("""
                SELECT r.ZFIRSTNAME, r.ZLASTNAME, p.ZFULLNUMBER
                FROM ZABCDRECORD r
                JOIN ZABCDPHONENUMBER p ON p.ZOWNER = r.ROWID
                WHERE r.ZFIRSTNAME IS NOT NULL
            """)
            for row in cur.fetchall():
                name = f"{row[0] or ''} {row[1] or ''}".strip()
                if name and row[2]:
                    clean = re.sub(r'[\s()\-+]', '', row[2])
                    if len(clean) >= 10:
                        _contact_cache[clean[-10:]] = name
            conn.close()
        _contact_cache_ts = time.time()
        log.info("Loaded %d contacts (fallback)", len(_contact_cache))
    except Exception as e:
        log.warning("Fallback contacts failed: %s", e)


def resolve_contact(phone_or_email: str) -> str:
    """Resolve phone/email to contact name."""
    if not phone_or_email or phone_or_email == "Unknown":
        return "Unknown"
    _load_contacts()
    clean = re.sub(r'[\s()\-+]', '', phone_or_email)
    if len(clean) >= 10:
        name = _contact_cache.get(clean[-10:])
        if name:
            return name
    # Direct lookup by email
    name = _contact_cache.get(phone_or_email)
    if name:
        return name
    # Show last 4 digits instead of full number
    if phone_or_email.startswith("+") and len(phone_or_email) > 6:
        return f"...{phone_or_email[-4:]}"
    return phone_or_email


def find_phone_for_name(name: str) -> str | None:
    """Look up a phone number by contact name. Prefers exact first-name match."""
    _load_contacts()
    name_lower = name.lower()

    # Pass 1: exact first-name match (highest confidence)
    for phone, contact_name in _contact_cache.items():
        first = contact_name.lower().split()[0] if contact_name else ""
        if first == name_lower or name_lower == contact_name.lower():
            return f"+1{phone}" if len(phone) == 10 else phone

    # Pass 2: first name starts with query
    for phone, contact_name in _contact_cache.items():
        first = contact_name.lower().split()[0] if contact_name else ""
        if first.startswith(name_lower) or name_lower.startswith(first):
            if len(name_lower) >= 4 and len(first) >= 4:  # avoid short false matches
                return f"+1{phone}" if len(phone) == 10 else phone

    # Pass 3: one of them contains the other (handles Samyukta/samyuktha)
    for phone, contact_name in _contact_cache.items():
        first = contact_name.lower().split()[0] if contact_name else ""
        if len(name_lower) >= 5 and len(first) >= 5:
            # Check if one contains the other (minus last 1-2 chars)
            short_query = name_lower[:-1]  # "samyukt" from "samyukta"
            short_name = first[:-1]        # "samyukth" from "samyuktha"
            if short_query in first or short_name in name_lower or short_query == short_name:
                return f"+1{phone}" if len(phone) == 10 else phone

    # Pass 4: substring match (least confident)
    for phone, contact_name in _contact_cache.items():
        if name_lower in contact_name.lower():
            return f"+1{phone}" if len(phone) == 10 else phone

    return None


# ── Clean Text for Speech ──────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Clean message text — remove emojis, URLs, special chars."""
    if not text:
        return ""
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
                  r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0000FE00-\U0000FEFF'
                  r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
                  r'\U00002600-\U000026FF\uFFFC]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:200]


# ── Read Messages ──────────────────────────────────────────────────────────

def get_recent_messages(count: int = 5) -> dict:
    """Get most recent messages across all conversations."""
    if not os.path.exists(IMESSAGE_DB):
        return {"ok": False, "error": "Messages database not found."}
    try:
        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COALESCE(h.id, 'Unknown'), m.text,
                   datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime'),
                   m.is_from_me
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC LIMIT ?
        """, (count,))
        messages = []
        for row in cursor.fetchall():
            text = _clean_text(row[1])
            if not text:
                continue
            messages.append({
                "from": "me" if row[3] else resolve_contact(row[0]),
                "text": text,
                "time": row[2] or "",
            })
        conn.close()
        log.info("Got %d recent messages", len(messages))
        return {"ok": True, "messages": messages}
    except sqlite3.OperationalError as e:
        if "unable to open" in str(e).lower():
            return {"ok": False, "error": "Grant Full Disk Access to Terminal in System Settings > Privacy."}
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_messages_from(contact: str, count: int = 5) -> dict:
    """Get recent messages from a specific contact (fuzzy name match)."""
    if not os.path.exists(IMESSAGE_DB):
        return {"ok": False, "error": "Messages database not found."}

    # Resolve contact name to phone number
    phone = find_phone_for_name(contact)

    try:
        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Search by: phone number (if found), display_name, or handle
        search_terms = [f"%{contact}%"]
        if phone:
            clean_phone = re.sub(r'[\s()\-+]', '', phone)
            search_terms.append(f"%{clean_phone[-10:]}%")
        if len(contact) > 3:
            search_terms.append(f"%{contact[:4]}%")

        where = " OR ".join(["h.id LIKE ?" for _ in search_terms] +
                            ["c.display_name LIKE ?" for _ in search_terms])
        params = search_terms + search_terms + [count]

        cursor.execute(f"""
            SELECT COALESCE(h.id, 'Unknown'), m.text,
                   datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime'),
                   m.is_from_me
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text IS NOT NULL AND m.text != '' AND ({where})
            ORDER BY m.date DESC LIMIT ?
        """, params)

        messages = []
        for row in cursor.fetchall():
            text = _clean_text(row[1])
            if not text:
                continue
            messages.append({
                "from": "me" if row[3] else resolve_contact(row[0]),
                "text": text,
                "time": row[2] or "",
            })
        conn.close()

        if not messages and phone:
            # Try direct phone search as last resort
            return _search_by_phone(phone, count)

        return {"ok": True, "contact": contact, "messages": messages}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _search_by_phone(phone: str, count: int = 5) -> dict:
    """Direct phone number search in messages."""
    try:
        clean = re.sub(r'[\s()\-+]', '', phone)[-10:]
        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COALESCE(h.id, 'Unknown'), m.text,
                   datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime'),
                   m.is_from_me
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND m.text != '' AND h.id LIKE ?
            ORDER BY m.date DESC LIMIT ?
        """, (f"%{clean}%", count))
        messages = []
        for row in cursor.fetchall():
            text = _clean_text(row[1])
            if not text:
                continue
            messages.append({
                "from": "me" if row[3] else resolve_contact(row[0]),
                "text": text,
                "time": row[2] or "",
            })
        conn.close()
        return {"ok": True, "messages": messages}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Send Messages ──────────────────────────────────────────────────────────

def send_message(to: str, text: str) -> dict:
    """Send an iMessage. Tries Shortcuts first, then AppleScript fallback."""
    if not to or not text:
        return {"ok": False, "error": "Need both recipient and message."}

    # Resolve name to phone number if needed
    if not re.match(r'^[\+\d]', to):
        phone = find_phone_for_name(to)
        if phone:
            log.info("Resolved '%s' to %s", to, phone)
            to = phone
        else:
            return {"ok": False, "error": f"Can't find phone number for '{to}'. Try using their number directly."}

    # Clean inputs
    to_clean = re.sub(r'[^a-zA-Z0-9@.+\- ]', '', to)
    text_clean = text.replace('"', '\\"').replace("'", "\\'")

    # Method 1: Shortcuts (if "Send Message" shortcut exists)
    try:
        result = subprocess.run(
            ["shortcuts", "run", "Send Message", "-i", f"{to_clean}\n{text_clean}"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            log.info("Sent via Shortcuts to %s", to_clean)
            return {"ok": True, "to": to_clean, "method": "shortcuts"}
    except Exception:
        pass

    # Method 2: AppleScript
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
            log.info("Sent via AppleScript to %s", to_clean)
            return {"ok": True, "to": to_clean, "method": "applescript"}
        else:
            error = result.stderr.strip()[:100]
            log.warning("AppleScript send failed: %s", error)
            return {"ok": False, "error": f"Failed to send: {error}. Is Messages.app open?"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Messages.app took too long to respond."}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── New Message Detection ──────────────────────────────────────────────────

_last_msg_check: float = 0
_last_msg_rowid: int = 0


def check_new_messages() -> list[dict]:
    """Check for new messages since last check. For polling (call every 30s)."""
    global _last_msg_check, _last_msg_rowid

    if not os.path.exists(IMESSAGE_DB):
        return []

    try:
        # Quick check: has the file been modified?
        mtime = os.path.getmtime(IMESSAGE_DB)
        if mtime <= _last_msg_check:
            return []
        _last_msg_check = mtime

        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        cursor = conn.cursor()

        if _last_msg_rowid == 0:
            # First run: get current max ROWID
            cursor.execute("SELECT MAX(ROWID) FROM message")
            row = cursor.fetchone()
            _last_msg_rowid = row[0] if row and row[0] else 0
            conn.close()
            return []

        cursor.execute("""
            SELECT COALESCE(h.id, 'Unknown'), m.text,
                   datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime'),
                   m.is_from_me, m.ROWID
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.ROWID > ? AND m.text IS NOT NULL AND m.text != '' AND m.is_from_me = 0
            ORDER BY m.date ASC
        """, (_last_msg_rowid,))

        new_msgs = []
        for row in cursor.fetchall():
            text = _clean_text(row[1])
            if text:
                new_msgs.append({
                    "from": resolve_contact(row[0]),
                    "text": text,
                    "time": row[2] or "",
                })
            if row[4] > _last_msg_rowid:
                _last_msg_rowid = row[4]

        conn.close()
        if new_msgs:
            log.info("Detected %d new messages", len(new_msgs))
        return new_msgs

    except Exception as e:
        log.debug("New message check failed: %s", e)
        return []
