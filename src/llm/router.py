"""LLM router: routes each request across several (mostly free-tier) providers.

Per request it picks the first *available* provider in the task's preference order and fails over to
the next one on error. Per provider it tracks:

* a sliding-window rate budget (requests and tokens per 60 s), which may be SHARED by several slots
  (all OpenRouter free models draw on one account-wide limit);
* a 429 cooldown taken from the provider's Retry-After header;
* a circuit breaker: CLOSED -> (N consecutive failures) -> OPEN -> (recovery time) -> HALF_OPEN,
  where exactly one probe request is allowed; success closes it, failure re-opens it with a doubled
  recovery time (capped).

Streaming fails over only before the first token: once text has reached the user, switching model
mid-answer would produce an incoherent reply, so the error is surfaced instead.

State is in-process. That is right for a single instance (the free-tier deployment); running several
replicas would need the windows/breakers in a shared store (e.g. Redis) - see ADR 0004.
"""

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import structlog

from src.core.exceptions import LLMError
from src.llm.base import ErrorKind, LLMAdapter, LLMRequest, LLMResult, ProviderError, Usage

logger = structlog.get_logger(__name__)

WINDOW_SECONDS = 60.0

# Failures that say something about the provider's health (they trip the breaker).
_PENALISED = {ErrorKind.TRANSIENT, ErrorKind.BAD_REQUEST, ErrorKind.EMPTY, ErrorKind.INVALID_OUTPUT}


class LLMUnavailableError(LLMError):
    """No provider could serve the request now. Carries a Retry-After hint for the API client."""

    status_code = 503

    def __init__(self, message: str, retry_after: float, errors: list[ProviderError]):
        super().__init__(message, code="LLM_UNAVAILABLE", metadata={"retry_after": retry_after})
        self.retry_after = retry_after
        self.errors = errors


class LLMRefusedError(LLMError):
    status_code = 422

    def __init__(self, message: str):
        super().__init__(message, code="LLM_REFUSED")


class LLMStreamInterruptedError(LLMError):
    """The provider failed after tokens were already sent; failover would garble the answer."""

    def __init__(self, message: str):
        super().__init__(message, code="LLM_STREAM_INTERRUPTED")


class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RateWindow:
    """Sliding 60 s window of requests and tokens, shared by every slot in one rate group."""

    rpm_limit: int | None = None
    tpm_limit: int | None = None
    requests: deque = field(default_factory=deque)  # timestamps
    tokens: deque = field(default_factory=deque)  # [timestamp, tokens] (mutable for reconcile)
    cooldown_until: float = 0.0  # set by a 429 - limits are per account, so per group

    def _evict(self, now: float) -> None:
        while self.requests and now - self.requests[0] >= WINDOW_SECONDS:
            self.requests.popleft()
        while self.tokens and now - self.tokens[0][0] >= WINDOW_SECONDS:
            self.tokens.popleft()

    def tokens_used(self) -> int:
        return sum(entry[1] for entry in self.tokens)

    def blocked_for(self, now: float, est_tokens: int) -> float:
        """0 if a request of `est_tokens` fits now, else seconds until it (probably) will."""
        self._evict(now)
        waits = [max(0.0, self.cooldown_until - now)]
        if self.rpm_limit is not None and len(self.requests) >= self.rpm_limit:
            waits.append(WINDOW_SECONDS - (now - self.requests[0]))
        if self.tpm_limit is not None and self.tokens_used() + est_tokens > self.tpm_limit:
            # A request bigger than the whole budget still gets one try once the window is empty.
            if self.tokens:
                waits.append(WINDOW_SECONDS - (now - self.tokens[0][0]))
        return max(waits)

    def reserve(self, now: float, est_tokens: int) -> list:
        self.requests.append(now)
        entry = [now, est_tokens]
        self.tokens.append(entry)
        return entry


@dataclass
class ProviderLimits:
    rpm: int | None = None
    tpm: int | None = None
    group: str | None = None  # slots with the same group share one RateWindow


@dataclass
class ProviderSlot:
    adapter: LLMAdapter
    window: RateWindow
    group: str
    breaker: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    recovery_seconds: float = 0.0
    probe_in_flight: bool = False
    # stats (for /admin/llm and metrics)
    successes: int = 0
    failures: int = 0
    rate_limited: int = 0
    last_error: str | None = None
    last_latency_ms: float | None = None

    @property
    def name(self) -> str:
        return self.adapter.name


@dataclass
class StreamEvent:
    kind: Literal["start", "token", "done"]
    text: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage | None = None


class LLMRouter:
    def __init__(
        self,
        slots: list[tuple[LLMAdapter, ProviderLimits]],
        task_orders: dict[str, list[str]] | None = None,
        *,
        failure_threshold: int = 3,
        recovery_seconds: float = 120.0,
        max_recovery_seconds: float = 1800.0,
        call_timeout: float = 60.0,
        first_token_timeout: float = 20.0,
        idle_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        observer: Callable[..., None] | None = None,
    ):
        if not slots:
            raise ValueError("LLMRouter needs at least one provider")
        windows: dict[str, RateWindow] = {}
        self.slots: list[ProviderSlot] = []
        for adapter, limits in slots:
            group = limits.group or adapter.name
            window = windows.setdefault(group, RateWindow(limits.rpm, limits.tpm))
            self.slots.append(
                ProviderSlot(adapter, window, group, recovery_seconds=recovery_seconds)
            )
        self.task_orders = task_orders or {}
        self.failure_threshold = failure_threshold
        self.base_recovery = recovery_seconds
        self.max_recovery = max_recovery_seconds
        self.call_timeout = call_timeout
        self.first_token_timeout = first_token_timeout
        self.idle_timeout = idle_timeout
        self._clock = clock
        self._observer = observer
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- selection

    def order_for(self, task: str) -> list[ProviderSlot]:
        """Slots in preference order for a task. Order entries may be exact names or prefixes
        ("openrouter" matches every "openrouter:<model>" slot). Unlisted slots follow, mock last."""
        prefs = self.task_orders.get(task) or self.task_orders.get("default") or []
        ordered: list[ProviderSlot] = []
        for pref in prefs:
            for slot in self.slots:
                if slot not in ordered and (slot.name == pref or slot.name.startswith(pref + ":")):
                    ordered.append(slot)
        rest = [s for s in self.slots if s not in ordered]
        rest.sort(key=lambda s: s.name == "mock")  # the offline mock is always the last resort
        return ordered + rest

    def _unavailable_for(self, slot: ProviderSlot, now: float, est_tokens: int) -> float:
        """0 if the slot can take a request now, else seconds until it can."""
        if slot.breaker is BreakerState.OPEN:
            reopen_at = slot.opened_at + slot.recovery_seconds
            if now < reopen_at:
                return reopen_at - now
            slot.breaker = BreakerState.HALF_OPEN  # recovery time elapsed: allow one probe
            logger.info("LLM provider half-open, probing", provider=slot.name)
        if slot.breaker is BreakerState.HALF_OPEN and slot.probe_in_flight:
            return 1.0
        return slot.window.blocked_for(now, est_tokens)

    async def _acquire(self, request: LLMRequest, tried: set[str]) -> ProviderSlot | None:
        est = request.estimated_prompt_tokens() + request.max_tokens
        async with self._lock:
            now = self._clock()
            for slot in self.order_for(request.task):
                if slot.name in tried or self._unavailable_for(slot, now, est) > 0:
                    continue
                slot.window.reserve(now, est)
                if slot.breaker is BreakerState.HALF_OPEN:
                    slot.probe_in_flight = True
                tried.add(slot.name)
                return slot
        return None

    def _retry_after(self, request: LLMRequest) -> float:
        now = self._clock()
        est = request.estimated_prompt_tokens() + request.max_tokens
        waits = [self._unavailable_for(s, now, est) for s in self.order_for(request.task)]
        positive = [w for w in waits if w > 0]
        return round(min(positive), 1) if positive else 1.0

    # ------------------------------------------------------------ bookkeeping

    async def _on_success(self, slot: ProviderSlot, tokens: int | None, latency_ms: float) -> None:
        async with self._lock:
            if tokens is not None and slot.window.tokens:
                slot.window.tokens[-1][1] = tokens  # replace the estimate with real usage
            if slot.breaker is not BreakerState.CLOSED:
                logger.info("LLM provider recovered", provider=slot.name)
            slot.breaker = BreakerState.CLOSED
            slot.consecutive_failures = 0
            slot.recovery_seconds = self.base_recovery
            slot.probe_in_flight = False
            slot.successes += 1
            slot.last_latency_ms = latency_ms
        self._observe(slot, "success", latency_ms)

    async def _on_failure(self, slot: ProviderSlot, err: ProviderError) -> None:
        async with self._lock:
            now = self._clock()
            slot.failures += 1
            slot.last_error = f"{err.kind.value}: {err.message}"[:300]
            was_probe = slot.probe_in_flight
            slot.probe_in_flight = False
            if err.kind is ErrorKind.RATE_LIMITED:
                slot.rate_limited += 1
                wait = err.retry_after if err.retry_after is not None else 60.0
                slot.window.cooldown_until = max(slot.window.cooldown_until, now + wait)
                logger.warning("LLM provider rate-limited", provider=slot.name, cooldown_s=wait)
            elif err.kind is ErrorKind.AUTH:
                self._open(slot, now, self.max_recovery)  # a bad key will not fix itself soon
            elif err.kind in _PENALISED:
                slot.consecutive_failures += 1
                if was_probe:
                    self._open(slot, now, min(slot.recovery_seconds * 2, self.max_recovery))
                elif slot.consecutive_failures >= self.failure_threshold:
                    self._open(slot, now, slot.recovery_seconds)
            # REFUSAL: content-specific, says nothing about the provider's health.
        self._observe(slot, err.kind.value, None)

    def _open(self, slot: ProviderSlot, now: float, recovery: float) -> None:
        slot.breaker = BreakerState.OPEN
        slot.opened_at = now
        slot.recovery_seconds = recovery
        logger.error(
            "LLM provider circuit opened",
            provider=slot.name,
            recovery_s=recovery,
            last_error=slot.last_error,
        )

    async def _release(self, slot: ProviderSlot) -> None:
        """The caller abandoned the request (cancelled / stopped reading): free a probe slot."""
        async with self._lock:
            slot.probe_in_flight = False

    def _observe(self, slot: ProviderSlot, outcome: str, latency_ms: float | None) -> None:
        if self._observer:
            try:
                self._observer(provider=slot.name, outcome=outcome, latency_ms=latency_ms)
            except Exception:  # metrics must never break a request
                logger.exception("LLM observer failed")

    def _exhausted(self, request: LLMRequest, errors: list[ProviderError]) -> LLMError:
        if errors and all(e.kind is ErrorKind.REFUSAL for e in errors):
            return LLMRefusedError("Every provider declined this request.")
        tried = ", ".join(f"{e.provider} ({e.kind.value})" for e in errors) or "none available"
        return LLMUnavailableError(
            f"No LLM provider could serve the request. Tried: {tried}",
            retry_after=self._retry_after(request),
            errors=errors,
        )

    # --------------------------------------------------------------- public

    async def complete(
        self, request: LLMRequest, validate: Callable[[str], bool] | None = None
    ) -> LLMResult:
        """Non-streaming call with failover. `validate` rejects structurally unusable output
        (e.g. invalid JSON), which then counts as a provider failure and fails over."""
        tried: set[str] = set()
        errors: list[ProviderError] = []
        while (slot := await self._acquire(request, tried)) is not None:
            started = self._clock()
            settled = False
            try:
                try:
                    result = await asyncio.wait_for(
                        slot.adapter.complete(request), self.call_timeout
                    )
                    if validate is not None and not validate(result.text):
                        raise ProviderError(
                            slot.name, ErrorKind.INVALID_OUTPUT, "failed caller validation"
                        )
                except TimeoutError:
                    err = ProviderError(slot.name, ErrorKind.TRANSIENT, "call timed out")
                except ProviderError as e:
                    err = e
                except Exception as e:  # an adapter bug must not take the request down
                    err = ProviderError(slot.name, ErrorKind.TRANSIENT, repr(e))
                else:
                    latency = (self._clock() - started) * 1000
                    await self._on_success(slot, result.usage.total, latency)
                    settled = True
                    logger.info(
                        "LLM call served",
                        provider=slot.name,
                        task=request.task,
                        latency_ms=round(latency),
                    )
                    return result
                await self._on_failure(slot, err)
                settled = True
                errors.append(err)
                logger.warning("LLM provider failed, failing over", error=str(err))
            finally:
                if not settled:
                    await self._release(slot)
        raise self._exhausted(request, errors)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Yields start -> token* -> done. Fails over only until the first token is out."""
        tried: set[str] = set()
        errors: list[ProviderError] = []
        while (slot := await self._acquire(request, tried)) is not None:
            started_at = self._clock()
            agen = slot.adapter.stream(request).__aiter__()
            emitted = False
            settled = False
            usage: Usage | None = None
            try:
                try:
                    while True:
                        timeout = self.idle_timeout if emitted else self.first_token_timeout
                        try:
                            item = await asyncio.wait_for(agen.__anext__(), timeout)
                        except StopAsyncIteration:
                            break
                        if isinstance(item, Usage):
                            usage = item
                            continue
                        if not emitted:
                            emitted = True
                            yield StreamEvent("start", provider=slot.name, model=slot.adapter.model)
                        yield StreamEvent("token", text=item)
                    if not emitted:
                        raise ProviderError(slot.name, ErrorKind.EMPTY, "stream produced no text")
                except TimeoutError:
                    err = ProviderError(slot.name, ErrorKind.TRANSIENT, "stream timed out")
                except ProviderError as e:
                    err = e
                except Exception as e:
                    err = ProviderError(slot.name, ErrorKind.TRANSIENT, repr(e))
                else:
                    latency = (self._clock() - started_at) * 1000
                    await self._on_success(slot, usage.total if usage else None, latency)
                    settled = True
                    yield StreamEvent(
                        "done", provider=slot.name, model=slot.adapter.model, usage=usage
                    )
                    return
                await self._on_failure(slot, err)
                settled = True
                errors.append(err)
                if emitted:
                    raise LLMStreamInterruptedError(
                        f"{slot.name} failed mid-answer ({err.kind.value}); partial answer kept"
                    )
                logger.warning("LLM stream failed before first token, failing over", error=str(err))
            finally:
                if not settled:
                    await self._release(slot)
                try:
                    await agen.aclose()  # type: ignore[attr-defined]
                except Exception:
                    pass
        raise self._exhausted(request, errors)

    # ----------------------------------------------------------------- admin

    def status(self) -> dict:
        now = self._clock()
        out = {}
        for s in self.slots:
            w = s.window
            w._evict(now)
            out[s.name] = {
                "model": s.adapter.model,
                "rate_group": s.group,
                "breaker": s.breaker.value,
                "consecutive_failures": s.consecutive_failures,
                "cooldown_remaining_s": round(max(0.0, w.cooldown_until - now), 1),
                "rpm_used": len(w.requests),
                "rpm_limit": w.rpm_limit,
                "tpm_used": w.tokens_used(),
                "tpm_limit": w.tpm_limit,
                "successes": s.successes,
                "failures": s.failures,
                "rate_limited": s.rate_limited,
                "last_error": s.last_error,
                "last_latency_ms": round(s.last_latency_ms) if s.last_latency_ms else None,
            }
        return out

    @property
    def only_mock(self) -> bool:
        return all(s.name == "mock" for s in self.slots)

    async def aclose(self) -> None:
        for s in self.slots:
            try:
                await s.adapter.aclose()
            except Exception:
                logger.exception("closing LLM adapter failed", provider=s.name)
