"""LLMRouter behaviour, driven by scripted fake adapters and a fake clock (no network, no sleeps)."""

import asyncio

import pytest

from src.llm.base import ChatMessage, ErrorKind, LLMRequest, LLMResult, ProviderError, Usage
from src.llm.router import (
    BreakerState,
    LLMRefusedError,
    LLMRouter,
    LLMStreamInterruptedError,
    LLMUnavailableError,
    ProviderLimits,
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class Fake:
    """Scripted adapter. `script` items: "ok", an ErrorKind, ("rate", seconds), "hang", "boom"."""

    def __init__(self, name, script=None, text=None, tokens=None, stream_fail_after=None):
        self.name = name
        self.model = f"{name}-model"
        self.script = list(script or [])
        self.text = text or f"answer from {name}"
        self.tokens = tokens
        self.calls = 0
        self.stream_fail_after = stream_fail_after

    def _next(self):
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if isinstance(step, tuple) and step[0] == "rate":
            raise ProviderError(self.name, ErrorKind.RATE_LIMITED, "429", retry_after=step[1])
        if isinstance(step, ErrorKind):
            raise ProviderError(self.name, step, step.value)
        if step == "boom":
            raise RuntimeError("adapter bug")
        return step

    async def complete(self, request):
        step = self._next()
        if step == "hang":
            await asyncio.sleep(3600)
        return LLMResult(self.text, self.name, self.model, Usage(10, self.tokens))

    async def stream(self, request):
        step = self._next()
        if step == "hang":
            await asyncio.sleep(3600)
        for i, word in enumerate(self.text.split()):
            if self.stream_fail_after is not None and i == self.stream_fail_after:
                raise ProviderError(self.name, ErrorKind.TRANSIENT, "connection reset")
            yield word + " "
        yield Usage(10, self.tokens)

    async def aclose(self):
        pass


def req(task="default", max_tokens=100, text="hi"):
    return LLMRequest([ChatMessage("user", text)], task=task, max_tokens=max_tokens)


def make(*fakes, limits=None, orders=None, **kw):
    clock = kw.pop("clock", Clock())
    slots = [(f, (limits or {}).get(f.name, ProviderLimits())) for f in fakes]
    kw.setdefault("recovery_seconds", 100)
    kw.setdefault("max_recovery_seconds", 1000)
    kw.setdefault("call_timeout", 0.2)
    kw.setdefault("first_token_timeout", 0.2)
    kw.setdefault("idle_timeout", 0.2)
    return LLMRouter(slots, orders or {}, clock=clock, **kw), clock


# ------------------------------------------------------------------ routing order


async def test_uses_the_first_provider_in_the_task_order():
    a, b = Fake("a"), Fake("b")
    router, _ = make(a, b, orders={"tutor": ["b", "a"], "default": ["a", "b"]})
    assert (await router.complete(req("tutor"))).provider == "b"
    assert (await router.complete(req("other"))).provider == "a"  # falls back to "default"


def test_order_entries_match_name_prefixes_and_mock_goes_last():
    mock, o1, o2, g = Fake("mock"), Fake("openrouter:x"), Fake("openrouter:y"), Fake("groq")
    router, _ = make(mock, o1, o2, g, orders={"default": ["openrouter"]})
    assert [s.name for s in router.order_for("default")] == [
        "openrouter:x",
        "openrouter:y",
        "groq",
        "mock",
    ]


# ----------------------------------------------------------------------- failover


async def test_transient_error_fails_over_to_the_next_provider():
    a, b = Fake("a", [ErrorKind.TRANSIENT]), Fake("b")
    router, _ = make(a, b, orders={"default": ["a", "b"]})
    result = await router.complete(req())
    assert result.provider == "b"
    assert router.status()["a"]["consecutive_failures"] == 1


async def test_each_provider_is_tried_at_most_once_per_request():
    a, b = Fake("a", [ErrorKind.TRANSIENT] * 5), Fake("b", [ErrorKind.TRANSIENT] * 5)
    router, _ = make(a, b)
    with pytest.raises(LLMUnavailableError) as exc:
        await router.complete(req())
    assert (a.calls, b.calls) == (1, 1)
    assert {e.provider for e in exc.value.errors} == {"a", "b"}


async def test_adapter_bug_is_contained_and_fails_over():
    router, _ = make(Fake("a", ["boom"]), Fake("b"))
    assert (await router.complete(req())).provider == "b"


async def test_a_hanging_provider_times_out_and_fails_over():
    router, _ = make(Fake("a", ["hang"]), Fake("b"))
    assert (await router.complete(req())).provider == "b"


async def test_validation_failure_fails_over():
    """A 200 with unusable content (e.g. invalid JSON) must not be returned to the caller."""
    a, b = Fake("a", text="not json"), Fake("b", text='{"ok": 1}')
    router, _ = make(a, b)
    result = await router.complete(req(), validate=lambda t: t.startswith("{"))
    assert result.provider == "b"


# --------------------------------------------------------------------- rate limits


async def test_429_cools_the_provider_down_for_retry_after_seconds():
    a, b = Fake("a", [("rate", 30)]), Fake("b")
    router, clock = make(a, b)
    assert (await router.complete(req())).provider == "b"
    assert (await router.complete(req())).provider == "b"  # a still cooling down
    clock.advance(31)
    assert (await router.complete(req())).provider == "a"  # recovered, no breaker penalty
    assert router.status()["a"]["breaker"] == "closed"


async def test_rpm_budget_uses_a_sliding_window():
    a, b = Fake("a"), Fake("b")
    router, clock = make(a, b, limits={"a": ProviderLimits(rpm=2)})
    assert [(await router.complete(req())).provider for _ in range(3)] == ["a", "a", "b"]
    clock.advance(59)
    assert (await router.complete(req())).provider == "b"  # window not yet slid
    clock.advance(2)
    assert (await router.complete(req())).provider == "a"


async def test_tpm_budget_blocks_large_requests_and_reconciles_actual_usage():
    a, b = Fake("a", tokens=5), Fake("b")
    router, _ = make(a, b, limits={"a": ProviderLimits(tpm=500)})
    assert (await router.complete(req(max_tokens=300))).provider == "a"  # reserved ~301, used 15
    assert (await router.complete(req(max_tokens=300))).provider == "a"  # fits after reconcile
    assert (await router.complete(req(max_tokens=2000))).provider == "b"  # can never fit now


async def test_slots_in_one_rate_group_share_a_single_budget():
    """OpenRouter free models share one account limit: two slots must not double it."""
    x, y, z = Fake("openrouter:x"), Fake("openrouter:y"), Fake("groq")
    shared = ProviderLimits(rpm=2, group="openrouter")
    router, _ = make(x, y, z, limits={"openrouter:x": shared, "openrouter:y": shared})
    served = [(await router.complete(req())).provider for _ in range(3)]
    assert served == ["openrouter:x", "openrouter:x", "groq"]  # y was never allowed to add 2 more


async def test_a_429_on_one_model_cools_down_its_whole_rate_group():
    x, y, z = Fake("openrouter:x", [("rate", 60)]), Fake("openrouter:y"), Fake("groq")
    shared = ProviderLimits(group="openrouter")
    router, _ = make(x, y, z, limits={"openrouter:x": shared, "openrouter:y": shared})
    assert (await router.complete(req())).provider == "groq"
    assert y.calls == 0


# ------------------------------------------------------------------ circuit breaker


async def test_breaker_opens_after_n_consecutive_failures():
    a, b = Fake("a", [ErrorKind.TRANSIENT] * 3), Fake("b")
    router, _ = make(a, b, failure_threshold=3)
    for _ in range(3):
        await router.complete(req())
    assert router.status()["a"]["breaker"] == "open"
    calls = a.calls
    await router.complete(req())
    assert a.calls == calls  # skipped while open


async def test_half_open_allows_exactly_one_probe_and_success_closes():
    a, b = Fake("a", [ErrorKind.TRANSIENT] * 3), Fake("b")
    router, clock = make(a, b, failure_threshold=3)
    for _ in range(3):
        await router.complete(req())
    clock.advance(101)
    assert (await router.complete(req())).provider == "a"  # the probe succeeded
    assert router.status()["a"]["breaker"] == "closed"


async def test_failed_probe_reopens_with_doubled_recovery():
    a, b = Fake("a", [ErrorKind.TRANSIENT] * 4), Fake("b")
    router, clock = make(a, b, failure_threshold=3)
    for _ in range(3):
        await router.complete(req())
    clock.advance(101)
    await router.complete(req())  # probe fails
    slot = router.slots[0]
    assert slot.breaker is BreakerState.OPEN and slot.recovery_seconds == 200
    clock.advance(150)
    await router.complete(req())
    assert a.calls == 4  # still open: recovery doubled to 200s


async def test_only_one_concurrent_probe_while_half_open():
    a, b = Fake("a", [ErrorKind.TRANSIENT] * 3 + ["hang"]), Fake("b")
    router, clock = make(a, b, failure_threshold=3, call_timeout=5)
    for _ in range(3):
        await router.complete(req())
    clock.advance(101)
    probe = asyncio.create_task(router.complete(req()))
    await asyncio.sleep(0.01)
    assert (await router.complete(req())).provider == "b"  # probe in flight -> not selected
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe
    assert router.slots[0].probe_in_flight is False  # a cancelled probe is released


async def test_auth_error_opens_the_breaker_immediately():
    router, _ = make(Fake("a", [ErrorKind.AUTH]), Fake("b"))
    await router.complete(req())
    assert router.status()["a"]["breaker"] == "open"


async def test_refusal_fails_over_without_penalising_the_provider():
    router, _ = make(Fake("a", [ErrorKind.REFUSAL]), Fake("b"))
    assert (await router.complete(req())).provider == "b"
    assert router.status()["a"]["consecutive_failures"] == 0


async def test_all_refusals_raise_a_refusal_not_an_outage():
    router, _ = make(Fake("a", [ErrorKind.REFUSAL]), Fake("b", [ErrorKind.REFUSAL]))
    with pytest.raises(LLMRefusedError):
        await router.complete(req())


async def test_when_everything_is_busy_it_fails_fast_with_a_retry_after_hint():
    a = Fake("a", [("rate", 40)])
    router, _ = make(a)
    with pytest.raises(LLMUnavailableError):
        await router.complete(req())
    with pytest.raises(LLMUnavailableError) as exc:
        await router.complete(req())
    assert a.calls == 1  # the cooling provider was not hammered
    assert 30 < exc.value.retry_after <= 40


# ------------------------------------------------------------------------ streaming


async def _collect(router, request):
    events = []
    async for ev in router.stream(request):
        events.append(ev)
    return events


async def test_stream_emits_start_tokens_done_with_provider():
    router, _ = make(Fake("a", text="one two three", tokens=3))
    events = await _collect(router, req())
    assert events[0].kind == "start" and events[0].provider == "a"
    assert "".join(e.text for e in events if e.kind == "token") == "one two three "
    assert events[-1].kind == "done" and events[-1].usage.output_tokens == 3


async def test_stream_fails_over_before_the_first_token():
    router, _ = make(Fake("a", [ErrorKind.TRANSIENT]), Fake("b", text="hello"))
    events = await _collect(router, req())
    assert events[0].provider == "b"


async def test_stream_first_token_timeout_fails_over():
    router, _ = make(Fake("a", ["hang"]), Fake("b", text="hello"))
    events = await _collect(router, req())
    assert events[0].provider == "b"


async def test_stream_never_fails_over_after_tokens_were_sent():
    """Switching model mid-answer would stitch two different answers together."""
    a = Fake("a", text="one two three four", stream_fail_after=2)
    b = Fake("b")
    router, _ = make(a, b)
    got = []
    with pytest.raises(LLMStreamInterruptedError):
        async for ev in router.stream(req()):
            got.append(ev)
    assert "".join(e.text for e in got if e.kind == "token") == "one two "
    assert b.calls == 0


async def test_a_consumer_that_stops_reading_releases_a_probe_slot():
    a, b = Fake("a", [ErrorKind.TRANSIENT] * 3, text="one two three"), Fake("b")
    router, clock = make(a, b, failure_threshold=3)
    for _ in range(3):
        await router.complete(req())
    clock.advance(101)
    gen = router.stream(req())
    first = await gen.__anext__()
    assert first.provider == "a"
    await gen.aclose()
    assert router.slots[0].probe_in_flight is False


# --------------------------------------------------------------------------- admin


async def test_status_reports_per_provider_state():
    router, _ = make(Fake("a"), limits={"a": ProviderLimits(rpm=30, tpm=8000)})
    await router.complete(req())
    s = router.status()["a"]
    assert s["rpm_used"] == 1 and s["rpm_limit"] == 30 and s["successes"] == 1
    assert s["breaker"] == "closed" and s["model"] == "a-model"


async def test_observer_receives_outcomes_and_its_errors_are_swallowed():
    seen = []

    def observer(**kw):
        seen.append(kw["outcome"])
        raise RuntimeError("metrics backend down")

    router, _ = make(Fake("a", [ErrorKind.TRANSIENT]), Fake("b"), observer=observer)
    await router.complete(req())
    assert seen == ["transient", "success"]
