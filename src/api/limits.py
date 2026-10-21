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
from fastapi import Request

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

    def peek(self, key: object) -> float | None:
        """Like `hit`, without recording: seconds until allowed if the key is at its limit."""
        now = self._clock()
        hits = self._hits.get(key)
        if not hits:
            return None
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        return self.window - (now - hits[0]) if len(hits) >= self.limit else None

    def reset(self, key: object) -> None:
        self._hits.pop(key, None)


def client_ip(request: Request) -> str:
    """Behind a proxy, run uvicorn with --proxy-headers so client.host is the real client."""
    return request.client.host if request.client else "unknown"


class AuthThrottle:
    """Brute-force protection for sign-in and sign-up, in process memory like the chat limiter.

    Only FAILED sign-ins count, and they are counted three ways so that one limit cannot be used
    against the others:
      * per (email, IP): a person who forgot a password locks out only themselves, and an attacker
        cannot lock a victim out from a different address;
      * per IP: one machine trying many accounts;
      * per email, high: guessing spread over many addresses.
    The check runs BEFORE the password is verified, so a locked-out client also cannot use
    bcrypt as a way to burn CPU. A success clears the (email, IP) count.
    """

    def __init__(self) -> None:
        window = settings.LOGIN_LOCKOUT_SECONDS
        self.pair = SlidingWindowLimiter(settings.LOGIN_MAX_FAILURES, window)
        self.ip = SlidingWindowLimiter(settings.LOGIN_IP_MAX_FAILURES, window)
        self.email = SlidingWindowLimiter(settings.LOGIN_EMAIL_MAX_FAILURES, 3600)
        self.signups = SlidingWindowLimiter(settings.REGISTER_MAX_PER_IP_PER_HOUR, 3600)

    def check_login(self, email: str, ip: str) -> None:
        waits = [
            self.pair.peek((email, ip)),
            self.ip.peek(ip),
            self.email.peek(email),
        ]
        wait = max((w for w in waits if w is not None), default=None)
        if wait is not None:
            metrics.RATE_LIMITED.labels("login").inc()
            logger.warning("Sign-in locked out", ip=ip)
            raise RateLimitError(
                "Too many failed sign-in attempts. Try again later.",
                retry_after=max(1, int(wait) + 1),
            )

    def login_failed(self, email: str, ip: str) -> None:
        self.pair.hit((email, ip))
        self.ip.hit(ip)
        self.email.hit(email)

    def login_succeeded(self, email: str, ip: str) -> None:
        self.pair.reset((email, ip))

    def check_register(self, ip: str) -> None:
        wait = self.signups.hit(ip)
        if wait is not None:
            metrics.RATE_LIMITED.labels("register").inc()
            raise RateLimitError(
                "Too many accounts created from this address. Try again later.",
                retry_after=max(1, int(wait) + 1),
            )


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
