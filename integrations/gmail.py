"""Gmail integration — draft, send, and search emails."""

import base64
from email.mime.text import MIMEText
from integrations.google_auth import get_service, api_error
from logger import get_logger

log = get_logger("gmail")


def _get_service(token_data: dict):
    return get_service(token_data, "gmail", "v1")


def _create_message(to: str, subject: str, body: str, sender: str = "me") -> dict:
    """Create a MIME email message."""
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return {"raw": raw}


def draft_email(token_data: dict, to: str, subject: str, body: str) -> dict:
    """Create an email draft (does NOT send).

    Returns:
        {"ok": True, "draft_id": str, "to": str, "subject": str}
    """
    service = _get_service(token_data)
    message = _create_message(to, subject, body)

    try:
        draft = service.users().drafts().create(
            userId="me",
            body={"message": message},
        ).execute()

        draft_id = draft["id"]
        log.info("Created draft: '%s' to %s (id: %s)", subject, to, draft_id)
        return {"ok": True, "draft_id": draft_id, "to": to, "subject": subject}
    except Exception as e:
        log.error("Gmail draft failed: %s", e)
        return api_error(e)


def send_email(token_data: dict, to: str, subject: str, body: str) -> dict:
    """Send an email immediately.

    Returns:
        {"ok": True, "message_id": str, "to": str, "subject": str}
    """
    service = _get_service(token_data)
    message = _create_message(to, subject, body)

    try:
        sent = service.users().messages().send(
            userId="me",
            body=message,
        ).execute()

        msg_id = sent["id"]
        log.info("Sent email: '%s' to %s (id: %s)", subject, to, msg_id)
        return {"ok": True, "message_id": msg_id, "to": to, "subject": subject}
    except Exception as e:
        log.error("Gmail send failed: %s", e)
        return api_error(e)


def list_emails(token_data: dict, query: str = "", max_results: int = 10) -> dict:
    """List recent emails from inbox (with optional search query).

    Returns:
        {"ok": True, "emails": [{"id": str, "subject": str, "from": str, "date": str, "snippet": str}]}
    """
    service = _get_service(token_data)
    try:
        params = {"userId": "me", "maxResults": min(max_results, 20)}
        if query:
            params["q"] = query
        result = service.users().messages().list(**params).execute()
        messages = result.get("messages", [])
        emails = []
        for msg_stub in messages[:max_results]:
            msg = service.users().messages().get(
                userId="me", id=msg_stub["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            emails.append({
                "id": msg["id"],
                "subject": headers.get("Subject", "(no subject)"),
                "from": headers.get("From", ""),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
            })
        log.info("Listed %d emails (query=%s)", len(emails), query or "inbox")
        return {"ok": True, "emails": emails}
    except Exception as e:
        log.error("Gmail list failed: %s", e)
        return api_error(e)


def get_email(token_data: dict, message_id: str) -> dict:
    """Get full email body by message ID.

    Returns:
        {"ok": True, "id": str, "subject": str, "from": str, "date": str, "body": str}
    """
    service = _get_service(token_data)
    try:
        msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        # Extract plain text body
        body = ""
        payload = msg.get("payload", {})
        if "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                    break
        elif payload.get("body", {}).get("data"):
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        return {
            "ok": True,
            "id": msg["id"],
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "body": body[:3000],  # cap body length for LLM context
        }
    except Exception as e:
        log.error("Gmail get failed: %s", e)
        return api_error(e)


def send_draft(token_data: dict, draft_id: str) -> dict:
    """Send a previously created draft.

    Returns:
        {"ok": True, "message_id": str}
    """
    service = _get_service(token_data)

    try:
        sent = service.users().drafts().send(
            userId="me",
            body={"id": draft_id},
        ).execute()

        msg_id = sent["id"]
        log.info("Sent draft %s (message: %s)", draft_id, msg_id)
        return {"ok": True, "message_id": msg_id}
    except Exception as e:
        log.error("Gmail send_draft failed: %s", e)
        return {"ok": False, "error": str(e)}
