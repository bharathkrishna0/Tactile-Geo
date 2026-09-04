from collections import defaultdict, deque
from time import monotonic

class InMemoryRateLimiter:
    """Small Milestone 1 guard; use a shared cache-backed limiter when deployed."""
    def __init__(self, limit: int = 20, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = monotonic()
        timestamps = self._requests[key]
        while timestamps and timestamps[0] <= now - self.window_seconds:
            timestamps.popleft()
        if len(timestamps) >= self.limit:
            return False
        timestamps.append(now)
        return True

upload_rate_limiter = InMemoryRateLimiter()
