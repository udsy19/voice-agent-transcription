"""Metrics collection — logs latency, errors, tool success rates.

Writes to DATA_DIR/metrics.jsonl for analysis.
"""

import json
import time
import threading
from collections import defaultdict
from config import DATA_DIR
from logger import get_logger

log = get_logger("metrics")

METRICS_FILE = DATA_DIR / "metrics.jsonl"


class MetricsCollector:
    def __init__(self):
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._summary: dict[str, list[float]] = defaultdict(list)

    def record(self, name: str, value: float, tags: dict = None):
        """Record a metric. Thread-safe."""
        entry = {"ts": time.time(), "name": name, "value": value, **(tags or {})}
        with self._lock:
            self._buffer.append(entry)
            self._summary[name].append(value)
            if len(self._buffer) >= 20:
                self._flush()

    def _flush(self):
        if not self._buffer:
            return
        try:
            with open(METRICS_FILE, "a") as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry) + "\n")
            self._buffer.clear()
        except Exception as e:
            log.debug("Metrics flush failed: %s", e)

    def get_summary(self) -> dict:
        """Get summary stats for display."""
        result = {}
        with self._lock:
            for name, values in self._summary.items():
                if not values:
                    continue
                result[name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                }
        return result

    def close(self):
        with self._lock:
            self._flush()


# Global instance
METRICS = MetricsCollector()
