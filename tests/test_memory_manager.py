"""MemoryManager: non-blocking submit, idempotency, caps, retries - over the in-memory store."""

import asyncio

import pytest

from src.memory.manager import MemoryEvent, MemoryManager
from src.memory.store import ConcurrentUpdate, InMemoryProfileStore


def _event(user_id=1, concepts=("lora",), intent="learn", event_id=None):
    event = MemoryEvent(user_id, "chat_turn", {"intent": intent, "concepts": list(concepts)})
    if event_id:
        event.event_id = event_id
    return event


@pytest.fixture
def store():
    return InMemoryProfileStore()


async def test_submitted_events_are_applied_by_the_worker(store):
    manager = MemoryManager(store)
    manager.start()
    assert manager.submit(_event()) is True
    await manager.drain()
    [row] = await store.get_mastery(1)
    assert row.concept == "lora" and row.evidence == 1
    assert manager.stats.applied == 1
    await manager.stop()


async def test_submit_never_blocks_even_with_no_worker(store):
    manager = MemoryManager(store, queue_size=1)
    assert manager.submit(_event()) is True
    assert manager.submit(_event()) is False  # queue full: dropped, not awaited
    assert manager.stats.dropped_full == 1


async def test_same_event_twice_is_applied_once(store):
    manager = MemoryManager(store)
    event = _event(event_id="11111111-1111-1111-1111-111111111111")
    assert await manager.process(event) is True
    assert await manager.process(event) is False
    [row] = await store.get_mastery(1)
    assert row.evidence == 1 and manager.stats.duplicates == 1


async def test_per_user_cap_limits_mutation_frequency(store):
    t = [0.0]
    manager = MemoryManager(store, per_minute=2, clock=lambda: t[0])
    assert [manager.submit(_event()) for _ in range(3)] == [True, True, False]
    assert manager.submit(_event(user_id=2)) is True  # other users are unaffected
    t[0] = 61.0
    assert manager.submit(_event()) is True  # window slid


async def test_transient_store_failure_is_retried(store):
    manager = MemoryManager(store, max_retries=2, base_backoff=0)
    real = store.apply_event
    calls = {"n": 0}

    async def flaky(*args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("db blip")
        return await real(*args)

    store.apply_event = flaky
    assert await manager.process(_event()) is True
    assert manager.stats.retried == 1


async def test_permanent_failure_is_counted_not_raised(store):
    manager = MemoryManager(store, max_retries=1, base_backoff=0)

    async def broken(*args):
        raise ConnectionError("db down")

    store.apply_event = broken
    assert await manager.process(_event()) is False
    assert manager.stats.failed == 1


async def test_concurrent_profile_edit_triggers_a_recompute(store):
    manager = MemoryManager(store, base_backoff=0)
    real = store.apply_event
    calls = {"n": 0}

    async def racing(*args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConcurrentUpdate("version moved")
        return await real(*args)

    store.apply_event = racing
    assert await manager.process(_event()) is True
    assert calls["n"] == 2


async def test_stale_snapshot_is_rejected_by_the_version_check(store):
    """Optimistic concurrency in the store itself: a profile write computed from an old version
    must not land."""

    def compute(snapshot):
        # simulate a user edit landing while this event was being computed
        store.profiles[1].version += 1
        from src.memory.model import Changes

        return Changes(profile_patch={"inferred_interests": {"lora": 1.0}})

    await store.get_profile(1)
    with pytest.raises(ConcurrentUpdate):
        await store.apply_event(1, "e1", "chat_turn", {}, compute)
    assert "e1" not in store.events  # nothing applied: all or nothing


async def test_worker_survives_a_bad_event(store):
    manager = MemoryManager(store, max_retries=0)
    manager.start()
    manager.submit(MemoryEvent(1, "chat_turn", {"intent": "learn", "concepts": None}))  # broken
    manager.submit(_event())
    await manager.drain()
    assert (await store.get_mastery(1))[0].concept == "lora"
    await manager.stop()


async def test_stop_drains_the_queue(store):
    manager = MemoryManager(store)
    manager.start()
    for _ in range(5):
        manager.submit(_event(concepts=("rlhf",)))
    await manager.stop()
    assert (await store.get_mastery(1))[0].evidence == 5
    await asyncio.sleep(0)
