"""Google Calendar — full CRUD: create, list, update, delete events.

Supports: location, notes, attendees, Google Meet, timezone, multi-day.
"""

from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from logger import get_logger

log = get_logger("gcal")


def _get_service(token_data: dict):
    if "credentials" in token_data:
        creds = token_data["credentials"]
    else:
        creds = Credentials(
            token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
        )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _api_error(e) -> dict:
    err = str(e)
    if "accessNotConfigured" in err or "has not been used" in err:
        return {"ok": False, "error": "Google Calendar API not enabled. Enable at console.cloud.google.com."}
    return {"ok": False, "error": err}


def create_event(token_data: dict, summary: str, start_time: str,
                 end_time: str = "", description: str = "",
                 location: str = "", attendees: list[str] | None = None,
                 timezone: str = "America/New_York",
                 add_meet: bool = False) -> dict:
    """Create a calendar event with full details."""
    service = _get_service(token_data)

    try:
        start_dt = datetime.fromisoformat(start_time)
    except ValueError:
        return {"ok": False, "error": f"Invalid start_time: {start_time}"}

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time)
        except ValueError:
            end_dt = start_dt + timedelta(hours=1)
    else:
        end_dt = start_dt + timedelta(hours=1)

    event_body = {
        "summary": summary,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": timezone},
    }
    if description:
        event_body["description"] = description
    if location:
        event_body["location"] = location
    if attendees:
        event_body["attendees"] = [{"email": e.strip()} for e in attendees if e.strip()]
    if add_meet:
        event_body["conferenceData"] = {
            "createRequest": {
                "requestId": f"muse-{int(datetime.now().timestamp())}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        }

    try:
        extra = {"conferenceDataVersion": 1} if add_meet else {}
        event = service.events().insert(
            calendarId="primary", body=event_body,
            sendUpdates="all" if attendees else "none",
            **extra,
        ).execute()

        meet_link = ""
        if event.get("conferenceData", {}).get("entryPoints"):
            for ep in event["conferenceData"]["entryPoints"]:
                if ep.get("entryPointType") == "video":
                    meet_link = ep["uri"]
                    break

        log.info("Created: %s at %s (attendees=%d, meet=%s)",
                 summary, start_time, len(attendees or []), bool(meet_link))
        return {
            "ok": True,
            "event_id": event["id"],
            "link": event.get("htmlLink", ""),
            "meet_link": meet_link,
            "summary": event["summary"],
            "start": event["start"].get("dateTime", event["start"].get("date", "")),
            "attendees_notified": bool(attendees),
        }
    except Exception as e:
        log.error("create_event failed: %s", e)
        return _api_error(e)


def list_events(token_data: dict, days_ahead: int = 1, max_results: int = 15) -> dict:
    """List events with full details."""
    service = _get_service(token_data)

    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    time_min = today_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_max = (today_start + timedelta(days=max(1, days_ahead))).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        result = service.events().list(
            calendarId="primary",
            timeMin=time_min, timeMax=time_max,
            maxResults=max_results,
            singleEvents=True, orderBy="startTime",
        ).execute()

        events = []
        for item in result.get("items", []):
            ev = {
                "summary": item.get("summary", "(no title)"),
                "start": item["start"].get("dateTime", item["start"].get("date", "")),
                "end": item["end"].get("dateTime", item["end"].get("date", "")),
                "id": item["id"],
            }
            if item.get("location"):
                ev["location"] = item["location"]
            if item.get("description"):
                ev["notes"] = item["description"][:200]
            if item.get("attendees"):
                ev["attendees"] = [a["email"] for a in item["attendees"][:10]]
            if item.get("conferenceData", {}).get("entryPoints"):
                for ep in item["conferenceData"]["entryPoints"]:
                    if ep.get("entryPointType") == "video":
                        ev["meet_link"] = ep["uri"]
                        break
            events.append(ev)

        log.info("Listed %d events (next %d days)", len(events), days_ahead)
        return {"ok": True, "events": events}
    except Exception as e:
        log.error("list_events failed: %s", e)
        return _api_error(e)


def update_event(token_data: dict, event_id: str, **fields) -> dict:
    """Update an existing event. Pass only fields to change."""
    service = _get_service(token_data)

    try:
        existing = service.events().get(calendarId="primary", eventId=event_id).execute()

        if "summary" in fields:
            existing["summary"] = fields["summary"]
        if "description" in fields:
            existing["description"] = fields["description"]
        if "location" in fields:
            existing["location"] = fields["location"]
        if "start_time" in fields:
            tz = existing["start"].get("timeZone", "America/New_York")
            existing["start"] = {"dateTime": fields["start_time"], "timeZone": tz}
        if "end_time" in fields:
            tz = existing["end"].get("timeZone", "America/New_York")
            existing["end"] = {"dateTime": fields["end_time"], "timeZone": tz}
        if "attendees" in fields:
            existing["attendees"] = [{"email": e.strip()} for e in fields["attendees"]]

        updated = service.events().update(
            calendarId="primary", eventId=event_id, body=existing,
            sendUpdates="all" if fields.get("attendees") else "none",
        ).execute()

        log.info("Updated event: %s", event_id)
        return {"ok": True, "event_id": event_id, "summary": updated.get("summary", "")}
    except Exception as e:
        log.error("update_event failed: %s", e)
        return _api_error(e)


def delete_event(token_data: dict, event_id: str) -> dict:
    """Delete/cancel a calendar event."""
    service = _get_service(token_data)
    try:
        service.events().delete(calendarId="primary", eventId=event_id,
                                sendUpdates="all").execute()
        log.info("Deleted event: %s", event_id)
        return {"ok": True, "event_id": event_id}
    except Exception as e:
        log.error("delete_event failed: %s", e)
        return _api_error(e)
