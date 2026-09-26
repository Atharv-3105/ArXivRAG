"""MemoryManager: turns interaction signals into profile updates, off the request path (ADR 0008).

    request handler --submit()--> bounded asyncio.Queue --worker--> ProfileStore.apply_event
         (never waits)                (drops when full)      (retries, idempotent, audited)

Plan §8: "MemoryManager must be async. Never block the response generation loop." submit() is a
put_nowait: it cannot block, and a full queue or a user over the per-minute cap drops the signal
(logged and counted). Signals are soft evidence; losing one costs a slightly staler profile, while
blocking would cost the user latency.

Delivery is at-most-once across a crash (the queue is in memory) and exactly-once in effect while
the process lives (event-id idempotency + transactional apply). The durable upgrade is an outbox
table drained by the worker - see ADR 0008 "Revisit when".
"""

import asyncio
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from uuid import uuid4

import structlog

from src.core.config import settings
from src.memory.model import compute_update
from src.memory.store import ConcurrentUpdate, ProfileStore

logger = structlog.get_logger(__name__)


@dataclass
class MemoryEvent:
    user_id: int
    kind: str  # "chat_turn" | "feedback" | "quiz"
    payload: dict
    event_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class MemoryStats:
    submitted: int = 0
    applied: int = 0
    duplicates: int = 0
    dropped_full: int = 0
    dropped_rate: int = 0
    retried: int = 0
    failed: int = 0


class MemoryManager:
    def __init__(
        self,
        store: ProfileStore,
        *,
        queue_size: int | None = None,
        max_retries: int | None = None,
        per_minute: int | None = None,
        clock: Callable[[], float] | None = None,
        base_backoff: float = 0.2,
    ):
        self.store = store
        self.queue: asyncio.Queue[MemoryEvent] = asyncio.Queue(
            maxsize=queue_size or settings.MEMORY_QUEUE_SIZE
        )
        self.max_retries = settings.MEMORY_MAX_RETRIES if max_retries is None else max_retries
        self.per_minute = per_minute or settings.MEMORY_MAX_UPDATES_PER_MINUTE
        self._clock = clock or (lambda: asyncio.get_running_loop().time())
        self._recent: dict[int, deque[float]] = {}
        self._worker: asyncio.Task | None = None
        self.base_backoff = base_backoff
        self.stats = MemoryStats()

    # ------------------------------------------------------------ producer side

    def _over_cap(self, user_id: int) -> bool:
        """Caps mutation frequency per user (plan §4): a script hammering feedback cannot churn
        a profile, and one user cannot fill the shared queue."""
        now = self._clock()
        window = self._recent.setdefault(user_id, deque())
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return True
        window.append(now)
        return False

    def submit(self, event: MemoryEvent) -> bool:
        """Never blocks, never raises. False means the signal was dropped."""
        self.stats.submitted += 1
        if self._over_cap(event.user_id):
            self.stats.dropped_rate += 1
            logger.warning("Memory signal dropped: per-user cap", user_id=event.user_id)
            return False
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.stats.dropped_full += 1
            logger.error("Memory signal dropped: queue full", queue_size=self.queue.maxsize)
            return False
        return True

    # ------------------------------------------------------------ consumer side

    async def process(self, event: MemoryEvent) -> bool:
        """Apply one event with retries. True if it changed state, False if duplicate/failed."""
        compute = partial(
            _compute,
            kind=event.kind,
            payload=event.payload,
            now=event.created_at,
        )
        for attempt in range(self.max_retries + 1):
            try:
                applied = await self.store.apply_event(
                    event.user_id, event.event_id, event.kind, event.payload, compute
                )
            except ConcurrentUpdate:
                self.stats.retried += 1  # someone edited the profile: recompute and retry now
                continue
            except Exception as e:
                if attempt == self.max_retries:
                    self.stats.failed += 1
                    logger.error(
                        "Memory event failed permanently",
                        event_id=event.event_id,
                        kind=event.kind,
                        error=str(e),
                    )
                    return False
                self.stats.retried += 1
                delay = self.base_backoff * 2**attempt * (0.5 + random.random())  # jitter
                logger.warning("Memory event failed, retrying", attempt=attempt + 1, error=str(e))
                await asyncio.sleep(delay)
                continue
            if applied:
                self.stats.applied += 1
                logger.info("Memory event applied", event_id=event.event_id, kind=event.kind)
            else:
                self.stats.duplicates += 1
            return applied
        self.stats.failed += 1
        logger.error("Memory event gave up after concurrent updates", event_id=event.event_id)
        return False

    async def _run(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                await self.process(event)
            except Exception:  # the worker must survive anything
                logger.exception("Memory worker error")
            finally:
                self.queue.task_done()

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="memory-worker")

    async def drain(self, wait_seconds: float = 5.0) -> None:
        """Wait for queued events (tests; graceful shutdown)."""
        await asyncio.wait_for(self.queue.join(), wait_seconds)

    async def stop(self, wait_seconds: float = 5.0) -> None:
        if self._worker is None:
            return
        try:
            await self.drain(wait_seconds)
        except TimeoutError:
            logger.warning("Memory queue not drained at shutdown", pending=self.queue.qsize())
        self._worker.cancel()
        self._worker = None


def _compute(snapshot, *, kind, payload, now):
    return compute_update(
        snapshot,
        kind,
        payload,
        now,
        alpha=settings.MEMORY_EMA_ALPHA,
        half_life=settings.MEMORY_DECAY_HALF_LIFE_DAYS,
    )
