"""Location services — get current city/coordinates via macOS CoreLocation.

Caches location for 10 minutes. Falls back gracefully if permission not granted.
To enable: System Settings > Privacy & Security > Location Services > Terminal (or Muse)
"""

import time
import threading
from logger import get_logger

log = get_logger("location")

_location_cache: dict = {}  # {"lat": ..., "lon": ..., "city": ..., "state": ..., "country": ...}
_cache_ts: float = 0
_CACHE_TTL = 600  # 10 minutes
_lock = threading.Lock()


def get_location() -> dict | None:
    """Get current location. Returns dict with lat/lon/city/state or None.
    Cached for 10 minutes. Non-blocking — returns cached value if available."""
    global _location_cache, _cache_ts

    if _location_cache and time.time() - _cache_ts < _CACHE_TTL:
        return _location_cache

    with _lock:
        # Double-check after acquiring lock
        if _location_cache and time.time() - _cache_ts < _CACHE_TTL:
            return _location_cache

        result = _fetch_location()
        if result:
            _location_cache = result
            _cache_ts = time.time()
            return result

    return _location_cache if _location_cache else None


def get_city() -> str:
    """Get current city name. Returns 'Unknown' if not available."""
    loc = get_location()
    if loc:
        city = loc.get("city", "")
        state = loc.get("state", "")
        if city and state:
            return f"{city}, {state}"
        return city or state or "Unknown"
    return "Unknown"


def _fetch_location() -> dict | None:
    """Fetch location from CoreLocation. Blocks up to 5 seconds."""
    try:
        import CoreLocation
        from Foundation import NSRunLoop, NSDate
    except ImportError:
        log.debug("CoreLocation not available — install pyobjc-framework-CoreLocation")
        return None

    try:
        manager = CoreLocation.CLLocationManager.alloc().init()

        # Check if we have permission
        status = CoreLocation.CLLocationManager.authorizationStatus()
        if status == 2:  # denied
            log.debug("Location permission denied")
            return None

        # Request permission if not determined
        if status == 0:
            manager.requestWhenInUseAuthorization()
            # Run loop briefly to let auth change
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(1.0))

        manager.startUpdatingLocation()

        # Wait for location (up to 3 seconds)
        loc = None
        deadline = time.time() + 3
        while time.time() < deadline:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.3))
            loc = manager.location()
            if loc and loc.horizontalAccuracy() > 0:
                break

        manager.stopUpdatingLocation()

        if not loc:
            log.debug("No location available — grant permission in System Settings > Privacy > Location Services")
            return None

        lat = loc.coordinate().latitude
        lon = loc.coordinate().longitude
        result = {"lat": lat, "lon": lon}

        # Reverse geocode to get city name
        geocoder = CoreLocation.CLGeocoder.alloc().init()
        geo_result = {"done": False}

        def on_geocode(placemarks, error):
            if placemarks and len(placemarks) > 0:
                pm = placemarks[0]
                geo_result["city"] = str(pm.locality()) if pm.locality() else ""
                geo_result["state"] = str(pm.administrativeArea()) if pm.administrativeArea() else ""
                geo_result["country"] = str(pm.country()) if pm.country() else ""
            geo_result["done"] = True

        geocoder.reverseGeocodeLocation_completionHandler_(loc, on_geocode)

        # Wait for geocode (up to 3 seconds)
        deadline = time.time() + 3
        while not geo_result["done"] and time.time() < deadline:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))

        if geo_result.get("city"):
            result["city"] = geo_result["city"]
            result["state"] = geo_result.get("state", "")
            result["country"] = geo_result.get("country", "")
            log.info("Location: %s, %s (%.4f, %.4f)", result["city"], result["state"], lat, lon)
        else:
            log.info("Location: %.4f, %.4f (geocode failed)", lat, lon)

        return result

    except Exception as e:
        log.debug("Location fetch failed: %s", e)
        return None
