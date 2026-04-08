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
            try:
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
            finally:
                conn.close()
        _contact_cache_ts = time.time()
        log.info("Loaded %d contacts (fallback)", len(_contact_cache))
    except Exception as e:
        log.warning("Fallback contacts failed: %s", e)


def resolve_contact(phone_or_email: str) -> str:
    """Resolve phone/email to contact name. Title-cases for display."""
    if not phone_or_email or phone_or_email == "Unknown":
        return "Unknown"
    _load_contacts()
    clean = re.sub(r'[\s()\-+]', '', phone_or_email)
    if len(clean) >= 10:
        name = _contact_cache.get(clean[-10:])
        if name:
            return _title_case_name(name)
    # Direct lookup by email
    name = _contact_cache.get(phone_or_email)
    if name:
        return _title_case_name(name)
    # Show last 4 digits instead of full number
    if phone_or_email.startswith("+") and len(phone_or_email) > 6:
        return f"...{phone_or_email[-4:]}"
    return phone_or_email


def _title_case_name(name: str) -> str:
    """Title-case a name, handling edge cases like 'mcdonald' -> 'McDonald'."""
    if not name:
        return name
    # If already has mixed case (user intentionally set it), keep it
    if name != name.lower() and name != name.upper():
        return name
    return name.title()


def find_phone_for_name(name: str) -> str | None:
    """Look up a phone number by contact name. Prefers exact first-name match.
    Returns the FIRST match. For disambiguation, use find_contacts() instead."""
    matches = find_contacts(name, limit=1)
    if matches:
        return matches[0]["phone"]
    return None


def find_contacts(name: str, limit: int = 10) -> list[dict]:
    """Find contacts matching a name query. Returns multiple for disambiguation.
    Each result: {"name": "...", "phone": "+1...", "score": "exact|prefix|fuzzy|substring"}
    """
    _load_contacts()
    name_lower = name.lower().strip()
    name_parts = name_lower.split()
    results = []
    seen_phones = set()

    def _add(phone, contact_name, score):
        formatted = f"+1{phone}" if len(phone) == 10 else phone
        if formatted not in seen_phones:
            seen_phones.add(formatted)
            results.append({"name": _title_case_name(contact_name), "phone": formatted, "score": score})

    # Pass 1: exact full-name or first-name match
    for phone, contact_name in _contact_cache.items():
        cn_lower = contact_name.lower()
        cn_parts = cn_lower.split()
        first = cn_parts[0] if cn_parts else ""

        # Multi-word query: check if ALL query words appear in the contact name
        if len(name_parts) > 1:
            if all(w in cn_lower for w in name_parts):
                _add(phone, contact_name, "exact")
                continue

        # Single-word: exact first-name or full-name match
        if first == name_lower or name_lower == cn_lower:
            _add(phone, contact_name, "exact")

    # If multi-word query got exact matches, return early
    if results and len(name_parts) > 1:
        return results[:limit]

    # Pass 2: first name starts with query (or vice versa)
    if len(results) < limit:
        for phone, contact_name in _contact_cache.items():
            first = contact_name.lower().split()[0] if contact_name else ""
            if first.startswith(name_lower) or name_lower.startswith(first):
                if len(name_lower) >= 4 and len(first) >= 4:
                    _add(phone, contact_name, "prefix")

    # Pass 3: fuzzy match (handles Samyukta/samyuktha)
    if len(results) < limit:
        for phone, contact_name in _contact_cache.items():
            first = contact_name.lower().split()[0] if contact_name else ""
            if len(name_lower) >= 5 and len(first) >= 5:
                short_query = name_lower[:-1]
                short_name = first[:-1]
                if short_query in first or short_name in name_lower or short_query == short_name:
                    _add(phone, contact_name, "fuzzy")

    # Pass 4: substring match — but only on word boundaries, not inside compound names
    if len(results) < limit:
        for phone, contact_name in _contact_cache.items():
            cn_lower = contact_name.lower()
            cn_words = cn_lower.split()
            # Check if query matches any word in the contact name
            if name_lower in cn_words or any(w.startswith(name_lower) for w in cn_words):
                _add(phone, contact_name, "substring")
            # Also match if query is a significant suffix (e.g. "purdue" in "Charan Purdue")
            elif len(name_lower) >= 4 and cn_lower.endswith(name_lower):
                _add(phone, contact_name, "substring")

    return results[:limit]


# ── Reply Watcher ─────────────────────────────────────────────────────────

_reply_watchers: list[dict] = []  # {"phone": ..., "sent_at": ..., "rowid_after": ..., "contact": ...}

def watch_for_reply(phone: str, contact_name: str):
    """Start watching for a reply from this phone number after sending a text."""
    if not os.path.exists(IMESSAGE_DB):
        return
    try:
        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            cur.execute("SELECT MAX(ROWID) FROM message")
            row = cur.fetchone()
            max_rowid = row[0] if row and row[0] else 0
        finally:
            conn.close()
        _reply_watchers.append({
            "phone": phone, "contact": contact_name,
            "sent_at": time.time(), "rowid_after": max_rowid,
        })
        # Only track last 5 watchers
        while len(_reply_watchers) > 5:
            _reply_watchers.pop(0)
        log.info("Watching for reply from %s", contact_name)
    except Exception as e:
        log.debug("Reply watcher setup failed: %s", e)


def check_for_replies() -> list[dict]:
    """Check if any watched contacts have replied. Call periodically (every 15-30s).
    Returns list of {"contact": name, "text": message, "time": timestamp}.
    Removes watchers older than 10 minutes or once a reply is found."""
    if not _reply_watchers or not os.path.exists(IMESSAGE_DB):
        return []

    now = time.time()
    replies = []
    to_remove = []

    for i, w in enumerate(_reply_watchers):
        # Expire watchers after 10 minutes
        if now - w["sent_at"] > 600:
            to_remove.append(i)
            continue

        clean_phone = re.sub(r'[\s()\-+]', '', w["phone"])[-10:]
        conn = None
        try:
            conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("""
                SELECT m.text, datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime')
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.ROWID > ? AND m.is_from_me = 0
                  AND m.text IS NOT NULL AND m.text != ''
                  AND h.id LIKE ?
                ORDER BY m.date ASC LIMIT 1
            """, (w["rowid_after"], f"%{clean_phone}%"))
            row = cur.fetchone()
            if row:
                text = _clean_text(row[0])
                if text:
                    replies.append({
                        "contact": w["contact"],
                        "text": text,
                        "time": row[1] or "",
                        "phone": w["phone"],
                    })
                    to_remove.append(i)
                    log.info("Reply detected from %s: %s", w["contact"], text[:40])
        except Exception as e:
            log.debug("Reply check failed: %s", e)
        finally:
            if conn:
                conn.close()

    # Remove expired/replied watchers (reverse order to preserve indices)
    for i in sorted(to_remove, reverse=True):
        _reply_watchers.pop(i)

    return replies


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
    conn = None
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
        log.info("Got %d recent messages", len(messages))
        return {"ok": True, "messages": messages}
    except sqlite3.OperationalError as e:
        if "unable to open" in str(e).lower():
            return {"ok": False, "error": "Grant Full Disk Access to Terminal in System Settings > Privacy."}
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if conn:
            conn.close()


def get_messages_from(contact: str, count: int = 5) -> dict:
    """Get recent messages from a specific contact (fuzzy name match)."""
    if not os.path.exists(IMESSAGE_DB):
        return {"ok": False, "error": "Messages database not found."}

    # Resolve contact name to phone number
    phone = find_phone_for_name(contact)

    conn = None
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

        if not messages and phone:
            return _search_by_phone(phone, count)

        return {"ok": True, "contact": contact, "messages": messages}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if conn:
            conn.close()


def _search_by_phone(phone: str, count: int = 5) -> dict:
    """Direct phone number search in messages."""
    conn = None
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
        return {"ok": True, "messages": messages}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if conn:
            conn.close()


# ── Send Messages ──────────────────────────────────────────────────────────

def send_message(to: str, text: str) -> dict:
    """Send an iMessage with verification. Tries multiple methods until confirmed in chat.db."""
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

    # Snapshot: get current max ROWID before sending
    rowid_before = _get_max_rowid()

    # Try each send method, verify after each attempt
    methods = _build_send_methods(to_clean, text_clean, text)
    last_error = ""

    for method_name, send_fn in methods:
        try:
            success, err = send_fn()
            if not success:
                log.debug("Method '%s' returned error: %s", method_name, err[:80] if err else "")
                last_error = err or f"{method_name} failed"
                continue

            # Verify: check chat.db for the message (wait up to 5s — db can be slow)
            verified = _verify_message_sent(to_clean, text, rowid_before, timeout=5.0)
            if verified:
                log.info("VERIFIED sent via %s to %s", method_name, to_clean)
                return {"ok": True, "to": to_clean, "method": method_name, "verified": True}
            else:
                log.warning("Method '%s' claimed success but message NOT in chat.db — trying next", method_name)
                last_error = f"{method_name} claimed success but message wasn't delivered"
                continue

        except subprocess.TimeoutExpired:
            log.debug("Method '%s' timed out", method_name)
            last_error = f"{method_name} timed out"
        except Exception as e:
            log.debug("Method '%s' error: %s", method_name, e)
            last_error = str(e)

    # All methods failed
    return {"ok": False, "error": (
        f"Message to {to_clean} was NOT delivered. {last_error}. "
        "Make sure Messages.app is open and you're signed into iMessage."
    )}


def _build_send_methods(to_clean: str, text_escaped: str, text_raw: str) -> list:
    """Build ordered list of (name, send_fn) methods to try.
    Timeouts kept short (5s) so total time for all methods < 25s."""
    from urllib.parse import quote
    methods = []

    # Method 1: AppleScript via chat (most reliable for existing conversations)
    script_chat = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{to_clean}" of targetService
            send "{text_escaped}" to targetBuddy
        end tell
    '''
    def send_applescript_chat():
        r = subprocess.run(["osascript", "-e", script_chat], capture_output=True, text=True, timeout=5)
        return r.returncode == 0, r.stderr.strip()[:100]
    methods.append(("applescript-chat", send_applescript_chat))

    # Method 2: AppleScript via buddy/service
    script_buddy = f'''
        tell application "Messages"
            set targetService to 1st service whose service type = iMessage
            set theBuddy to buddy "{to_clean}" of targetService
            send "{text_escaped}" to theBuddy
        end tell
    '''
    def send_applescript_buddy():
        r = subprocess.run(["osascript", "-e", script_buddy], capture_output=True, text=True, timeout=5)
        return r.returncode == 0, r.stderr.strip()[:100]
    methods.append(("applescript-buddy", send_applescript_buddy))

    # Method 3: Shortcuts (if "Send Message" shortcut exists)
    def send_shortcuts():
        r = subprocess.run(
            ["shortcuts", "run", "Send Message", "-i", f"{to_clean}\n{text_raw}"],
            capture_output=True, text=True, timeout=8)
        return r.returncode == 0, r.stderr.strip()[:100]
    methods.append(("shortcuts", send_shortcuts))

    return methods


def _get_max_rowid() -> int:
    """Get current max message ROWID from chat.db."""
    if not os.path.exists(IMESSAGE_DB):
        return 0
    conn = None
    try:
        conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT MAX(ROWID) FROM message")
        row = cur.fetchone()
        return row[0] if row and row[0] else 0
    except Exception:
        return 0
    finally:
        if conn:
            conn.close()


def _verify_message_sent(to: str, text: str, rowid_before: int, timeout: float = 3.0) -> bool:
    """Check chat.db for a new outgoing message to `to` after `rowid_before`.
    Checks both `text` column and `attributedBody` blob (macOS 26+ stores text there).
    Polls every 0.5s up to `timeout` seconds."""
    if not os.path.exists(IMESSAGE_DB):
        return False

    clean_to = re.sub(r'[\s()\-+]', '', to)[-10:]
    check_words = set(text.lower().split()[:5])

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.5)
        conn = None
        try:
            conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True, timeout=2)
            cur = conn.cursor()
            # Get new outgoing messages — include those with empty text (macOS 26+)
            cur.execute("""
                SELECT m.text, h.id, m.attributedBody
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.ROWID > ? AND m.is_from_me = 1
                ORDER BY m.ROWID DESC LIMIT 5
            """, (rowid_before,))
            for row in cur.fetchall():
                handle = row[1] or ""
                # Check recipient match
                if not clean_to or clean_to not in re.sub(r'[\s()\-+]', '', handle):
                    continue

                # Get message text — try text column first, then attributedBody blob
                msg_text = (row[0] or "").lower()
                if not msg_text and row[2]:
                    msg_text = _extract_text_from_attributed_body(row[2]).lower()

                if not msg_text:
                    # Message exists to right person but no text extracted —
                    # still counts as sent on macOS 26+ where text column is empty
                    log.info("Verified in chat.db (empty text, attributedBody present) → %s", handle)
                    return True

                # Check content overlap
                msg_words = set(msg_text.split())
                overlap = check_words & msg_words
                if len(overlap) >= min(2, len(check_words)):
                    log.info("Verified in chat.db: '%s' → %s", msg_text[:40], handle)
                    return True
        except Exception as e:
            log.debug("Verify check error: %s", e)
        finally:
            if conn:
                conn.close()

    log.warning("Message NOT verified in chat.db after %.1fs", timeout)
    return False


def _extract_text_from_attributed_body(blob: bytes) -> str:
    """Extract plain text from NSAttributedString blob in attributedBody column.
    macOS 26+ stores message text here instead of the text column."""
    if not blob:
        return ""
    try:
        # The blob is a streamtyped NSAttributedString
        # Plain text is embedded as UTF-8 after NSString marker
        decoded = blob.decode('utf-8', errors='replace')
        # Find text between NSString marker and next control sequence
        chunks = re.findall(r'[\x20-\x7e]{3,}', decoded)
        # Filter out framework class names
        skip = {'streamtyped', 'NSAttributedString', 'NSObject', 'NSString',
                'NSDictionary', 'NSNumber', 'NSValue', 'NSMutableAttributedString'}
        text_chunks = [c for c in chunks if c not in skip and not c.startswith('__kIM')]
        if text_chunks:
            # The actual message text is usually the longest chunk or the one after NSString
            # Filter: take chunks that look like actual message text (not metadata keys)
            real_text = [c for c in text_chunks if not c.startswith('NS') and len(c) > 2]
            if real_text:
                return real_text[0].lstrip('+ ')  # strip leading '+ ' marker
        return ""
    except Exception:
        return ""


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
        try:
            cursor = conn.cursor()

            if _last_msg_rowid == 0:
                cursor.execute("SELECT MAX(ROWID) FROM message")
                row = cursor.fetchone()
                _last_msg_rowid = row[0] if row and row[0] else 0
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

            if new_msgs:
                log.info("Detected %d new messages", len(new_msgs))
            return new_msgs
        finally:
            conn.close()

    except Exception as e:
        log.debug("New message check failed: %s", e)
        return []
