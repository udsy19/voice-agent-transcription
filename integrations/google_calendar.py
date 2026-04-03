"""Google Calendar integration — create and list events."""

from datetime import datetime, timedelta, timezone
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from logger import get_logger

log = get_logger("gcal")


def _get_service(token_data: dict):
    """Build a Calendar API service from token data."""
    if "credentials" in token_data:
        creds = token_data["credentials"]
    else:
        creds = Credentials(
            token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
        )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def create_event(token_data: dict, summary: str, start_time: str,
                 end_time: str = "", description: str = "",
                 attendees: list[str] | None = None) -> dict:
    """Create a calendar event.

    Args:
        token_data: OAuth token dict from OAuthManager
        summary: Event title
        start_time: ISO 8601 datetime (e.g., "2026-04-04T15:00:00")
        end_time: ISO 8601 datetime. Defaults to 1 hour after start.
        description: Optional event description
        attendees: Optional list of email addresses

    Returns:
        {"ok": True, "event_id": str, "link": str, "summary": str, "start": str}
    """
    service = _get_service(token_data)

    # Parse start time and default end to +1 hour
    try:
        start_dt = datetime.fromisoformat(start_time)
    except ValueError:
        return {"ok": False, "error": f"Invalid start_time format: {start_time}"}

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time)
        except ValueError:
            end_dt = start_dt + timedelta(hours=1)
    else:
        end_dt = start_dt + timedelta(hours=1)

    event_body = {
        "summary": summary,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/New_York"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/New_York"},
    }
    if description:
        event_body["description"] = description
    if attendees:
        event_body["attendees"] = [{"email": e} for e in attendees]

    try:
        event = service.events().insert(calendarId="primary", body=event_body).execute()
        log.info("Created event: %s at %s", summary, start_time)
        return {
            "ok": True,
            "event_id": event["id"],
            "link": event.get("htmlLink", ""),
            "summary": event["summary"],
            "start": event["start"].get("dateTime", event["start"].get("date", "")),
        }
    except Exception as e:
        log.error("Calendar create_event failed: %s", e)
        return {"ok": False, "error": str(e)}


def list_events(token_data: dict, days_ahead: int = 1, max_results: int = 10) -> dict:
    """List upcoming calendar events.

    Args:
        token_data: OAuth token dict
        days_ahead: How many days ahead to look (default 1 = today)
        max_results: Max events to return

    Returns:
        {"ok": True, "events": [{"summary": str, "start": str, "end": str, "id": str}]}
    """
    service = _get_service(token_data)

    now = datetime.now(timezone.utc)
    time_min = now.isoformat() + "Z"
    time_max = (now + timedelta(days=max(1, days_ahead))).isoformat() + "Z"

    try:
        result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = []
        for item in result.get("items", []):
            events.append({
                "summary": item.get("summary", "(no title)"),
                "start": item["start"].get("dateTime", item["start"].get("date", "")),
                "end": item["end"].get("dateTime", item["end"].get("date", "")),
                "id": item["id"],
            })

        log.info("Listed %d events (next %d days)", len(events), days_ahead)
        return {"ok": True, "events": events}
    except Exception as e:
        log.error("Calendar list_events failed: %s", e)
        return {"ok": False, "error": str(e)}
