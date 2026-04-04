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
