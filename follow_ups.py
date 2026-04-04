"""Smart follow-up tracker — nudge when emails get no reply."""

import time
import safe_json
from config import DATA_DIR
from logger import get_logger

log = get_logger("followup")

FOLLOWUPS_PATH = str(DATA_DIR / "follow_ups.json")


def add(to: str, subject: str, message_id: str = "", threshold_hours: int = 72) -> dict:
    """Track a sent email for follow-up."""
    data = safe_json.load(FOLLOWUPS_PATH, {"items": []})
    data["items"].append({
        "to": to, "subject": subject, "message_id": message_id,
        "sent_at": time.time(), "threshold_hours": threshold_hours,
        "reminded": False,
    })
    # Cap at 50 tracked items
    if len(data["items"]) > 50:
        data["items"] = data["items"][-50:]
    safe_json.save(FOLLOWUPS_PATH, data)
    log.info("Tracking follow-up: %s → %s", subject[:30], to)
    return {"ok": True}


def get_pending_reminders() -> list[dict]:
    """Get emails that need follow-up (past threshold, not yet reminded)."""
    data = safe_json.load(FOLLOWUPS_PATH, {"items": []})
    now = time.time()
    reminders = []
    for item in data["items"]:
        if item.get("reminded"):
            continue
        hours = (now - item["sent_at"]) / 3600
        if hours >= item.get("threshold_hours", 72):
            reminders.append({
                "to": item["to"],
                "subject": item["subject"],
                "hours_ago": int(hours),
                "days_ago": int(hours / 24),
            })
    return reminders


def mark_reminded(to: str, subject: str):
    """Mark a follow-up as reminded so it doesn't repeat."""
    data = safe_json.load(FOLLOWUPS_PATH, {"items": []})
    for item in data["items"]:
        if item["to"] == to and item["subject"] == subject:
            item["reminded"] = True
    safe_json.save(FOLLOWUPS_PATH, data)


def get_all() -> list[dict]:
    """Get all tracked follow-ups."""
    data = safe_json.load(FOLLOWUPS_PATH, {"items": []})
    return data["items"]
