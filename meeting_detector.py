"""Meeting detector — polls for active meetings via app detection + calendar awareness.

Emits WebSocket events when a meeting is detected so the UI can prompt the user to record.
"""

import time
import threading
from datetime import datetime, timezone
from logger import get_logger
from utils import get_active_app, get_browser_url

log = get_logger("meeting_detector")

# Apps that indicate a video/voice call
MEETING_APPS = {"zoom.us", "FaceTime", "Webex", "Microsoft Teams", "Slack"}

# URLs that indicate a browser-based meeting
MEETING_URLS = ["meet.google.com", "zoom.us/j/", "teams.microsoft.com/l/meetup"]

POLL_INTERVAL = 15  # seconds


class MeetingDetector:
    def __init__(self, oauth_manager, emit_fn):
        self._oauth = oauth_manager
        self._emit = emit_fn
        self._polling = False
        self._thread = None
        self._last_detected = None  # avoid duplicate notifications
        self._last_detected_time = 0

    def start_polling(self):
        if self._polling:
            return
        self._polling = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("Meeting detector started (polling every %ds)", POLL_INTERVAL)

    def stop_polling(self):
        self._polling = False

    def _poll_loop(self):
        while self._polling:
            try:
                detected = self._check()
                if detected:
                    key = detected.get("app") or detected.get("url") or "calendar"
                    now = time.time()
                    # Don't re-notify for same meeting within 5 minutes
                    if key != self._last_detected or (now - self._last_detected_time) > 300:
                        self._last_detected = key
                        self._last_detected_time = now
                        self._emit(detected)
                        log.info("Meeting detected: %s", key)
            except Exception as e:
                log.debug("Detection poll error: %s", e)
            time.sleep(POLL_INTERVAL)

    def _check(self):
        """Check all detection sources, return event dict or None."""
        # 1. Check frontmost app
        app = self._check_app()
        if app:
            return app

        # 2. Check browser URL
        url = self._check_browser()
        if url:
            return url

        # 3. Check calendar for active meeting
        cal = self._check_calendar()
        if cal:
            return cal

        return None

    def _check_app(self):
        """Check if a meeting app is frontmost."""
        try:
            active = get_active_app()
            if active in MEETING_APPS:
                return {
                    "type": "meeting_detected",
                    "source": "app",
                    "app": active,
                    "title": f"{active} call",
                }
        except Exception:
            pass
        return None

    def _check_browser(self):
        """Check if browser is showing a meeting URL."""
        try:
            url = get_browser_url()
            if url:
                for pattern in MEETING_URLS:
                    if pattern in url:
                        return {
                            "type": "meeting_detected",
                            "source": "browser",
                            "url": url,
                            "app": "Google Meet" if "meet.google" in url else "Zoom" if "zoom.us" in url else "Teams",
                            "title": "Browser meeting",
                        }
        except Exception:
            pass
        return None

    def _check_calendar(self):
        """Check if a calendar event with a meeting link is happening now."""
        if not self._oauth:
            return None
        try:
            token = self._oauth.get_token("google")
            if not token:
                return None
            from integrations.google_calendar import list_events
            events = list_events(token, days=1)
            if not events.get("ok"):
                return None

            now = datetime.now(timezone.utc)
            for ev in events.get("events", []):
                try:
                    from utils import parse_iso
                    start = parse_iso(ev.get("start"))
                    end = parse_iso(ev.get("end"))
                    if not start or not end:
                        continue
                    if start <= now <= end and ev.get("meet_link"):
                        return {
                            "type": "meeting_detected",
                            "source": "calendar",
                            "app": "Google Meet",
                            "title": ev.get("summary", "Meeting"),
                            "meet_link": ev["meet_link"],
                            "event": ev,
                        }
                except (ValueError, KeyError):
                    continue
        except Exception as e:
            log.debug("Calendar check error: %s", e)
        return None
