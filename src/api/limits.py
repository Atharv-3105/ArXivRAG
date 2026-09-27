"""Free-tier protection: per-user request rate and daily message quota (ADR 0006).

Two limits, two stores, on purpose:
  * per-minute rate: a sliding window in process memory. It only has to survive a minute, a
    restart resetting it is harmless, and it costs no I/O on the hot path.
  * daily quota: counted from the messages table. It must survive restarts (free hosts restart
    often, and a restart must not hand out a fresh quota), and the count is already there.
Admins are exempt from both.
"""

import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import structlog

from src.core import metrics
from src.core.config import settings
from src.core.exceptions import RateLimitError
from src.storage.conversations import ConversationStore

logger = structlog.get_logger(__name__)


class SlidingWindowLimiter:
    """At most `limit` hits per `window` seconds per key. Unlike a fixed per-minute counter, it
    never allows 2x the limit across a window boundary."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: dict[object, deque[float]] = {}

    def hit(self, key: object) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until the next hit is allowed."""
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return self.window - (now - hits[0])
        hits.append(now)
        if len(self._hits) > 10_000:  # bound memory: forget idle keys
            self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window}
        return None


def utc_day_start(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


async def enforce_chat_limits(
    user_id: int,
    is_admin: bool,
    limiter: SlidingWindowLimiter,
    conversations: ConversationStore,
    now: datetime | None = None,
) -> None:
    if is_admin:
        return
    wait = limiter.hit(user_id)
    if wait is not None:
        metrics.RATE_LIMITED.labels("per_minute").inc()
        raise RateLimitError("Too many requests, slow down", retry_after=max(1, int(wait) + 1))

    now = now or datetime.now(UTC)
    start = utc_day_start(now)
    try:
        used = await conversations.count_user_messages_since(user_id, start)
    except Exception as e:  # fail open: the LLM router has its own budgets as a backstop
        logger.error("Could not check daily quota; allowing", user_id=user_id, error=str(e))
        return
    if used >= settings.FREE_DAILY_MESSAGES:
        metrics.RATE_LIMITED.labels("daily_quota").inc()
        reset_in = int((start + timedelta(days=1) - now).total_seconds()) + 1
        raise RateLimitError(
            f"Daily limit of {settings.FREE_DAILY_MESSAGES} messages reached; resets at 00:00 UTC",
            retry_after=reset_in,
            code="DAILY_QUOTA_EXCEEDED",
        )
