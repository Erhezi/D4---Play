"""In-memory per-user daily request rate limiter (MVP)."""

from __future__ import annotations

from datetime import date

from ai_export_builder.config import settings


class RateLimiter:
    """Track daily request counts per user. Resets each calendar day."""

    def __init__(self, daily_limit: int | None = None) -> None:
        self._limit = daily_limit or settings.daily_request_limit
        # {user_id: {date_str: count}}
        self._counts: dict[str, dict[str, int]] = {}

    def _today(self) -> str:
        return date.today().isoformat()

    def check(self, user_id: str) -> bool:
        """Return True if the user is still within the daily limit."""
        today = self._today()
        user_counts = self._counts.get(user_id, {})
        return user_counts.get(today, 0) < self._limit

    def remaining(self, user_id: str) -> int:
        """Return how many requests the user has left today."""
        today = self._today()
        used = self._counts.get(user_id, {}).get(today, 0)
        return max(0, self._limit - used)

    def increment(self, user_id: str) -> None:
        """Record one request for the user."""
        today = self._today()
        if user_id not in self._counts:
            self._counts[user_id] = {}
        user_day = self._counts[user_id]
        user_day[today] = user_day.get(today, 0) + 1

    def reset(self, user_id: str) -> None:
        """Reset a user's counter (for testing)."""
        self._counts.pop(user_id, None)


# Module-level singleton
rate_limiter = RateLimiter()
