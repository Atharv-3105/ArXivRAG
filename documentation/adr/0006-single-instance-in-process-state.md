# ADR 0006: Single instance, in-process shared state (no Redis yet)

- **Status:** Accepted (2026-09-26). This deviates from the plan's Redis layer (§3, §5).
- **Affects:** the LLM router's budgets and breakers (ADR 0004), the per-minute rate limiter (`src/api/limits.py`), the MemoryManager queue (`src/memory/manager.py`) and profile loading.

## Context

The plan puts Redis in front of Postgres for sessions, rate-limit counters and hot profiles. On free tiers, Redis is another service with its own tiny quota (for example, a capped number of commands per day) and its own latency, often in a different region from the app.

The deployment target is **one** web instance, since free hosts give one small container.

## Decision

In-process state lives in the web process wherever losing it on a restart is harmless:

| State | Where | Lost on restart? | Why that's OK |
|---|---|---|---|
| Per-minute rate window | in-process sliding window | yes | it only has to cover 60 s |
| LLM rpm/tpm windows, breakers | in-process (ADR 0004) | yes | providers enforce their own limits, and we re-learn in one call |
| Memory event queue | in-process `asyncio.Queue` | yes, queued events are lost | signals are soft evidence (ADR 0008) |
| Daily message quota | **Postgres** (count of today's messages) | no | a restart must not hand out a fresh quota |
| Profiles, mastery | **Postgres**, read per request | no | 2 primary-key reads, about 1–2 ms on a warm connection |
| Sessions | none: stateless JWT | n/a | the token is the session |

No profile cache: a cache adds invalidation bugs (a profile edit not visible on the next turn) to save a couple of milliseconds that are not the bottleneck. The LLM call is.

## Revisit when

- **More than one instance.** In-process windows then multiply the effective limits by N, and breakers learn separately. Move the windows to Redis (`INCR` + `EXPIRE`, or a Lua sliding window) and the memory queue to a durable outbox (ADR 0008).
- Profile reads show up in p95 latency (Phase 6 metrics). Then add a read-through cache with explicit invalidation on write.

## Consequences

- Good: zero extra services and zero extra network hops. Every limit that must be durable already is.
- Cost: horizontal scaling needs the Redis step first. The code isolates each piece of state behind one class (`SlidingWindowLimiter`, `RateWindow`, `MemoryManager`), so each swap is local.
