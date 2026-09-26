# ADR 0008: Learner model (EMA + decay + SM-2) and the async memory pipeline

- **Status:** Accepted (2026-09-26)
- **Code:** `src/memory/model.py` (pure maths), `src/memory/store.py` (transactions), `src/memory/manager.py` (queue and worker), `src/memory/concepts.py` (vocabulary), `src/api/routes/me.py`, `src/api/routes/chat.py`
- **Tests:** `tests/test_memory_model.py`, `tests/test_memory_manager.py`, `tests/test_profile_store_sql.py`, `tests/test_auth_and_me_api.py`

## Context

The plan (Phase 3) asks for profiles with explicit and inferred fields, concept mastery with EMA smoothing, recency decay, an explicit-override lock, spaced repetition, an audit trail, idempotent updates, capped mutation frequency, and an async MemoryManager that never blocks a response.

## Decision

### The model (pure functions, `model.py`)
- **Evidence, not guesses.** Each signal is an *observation* o ∈ [0, 1] with a *weight* w:

  | Signal | o | w | Reading |
  |---|---|---|---|
  | asked to explain (learn) | 0.25 | 0.3 | weak evidence of a gap |
  | asked to critique / compare | 0.7 | 0.3 | weak evidence of fluency |
  | "I understood" / "I'm confused" | 0.8 / 0.1 | 0.6 | direct, self-reported |
  | quiz right / wrong (Phase 5) | 1 / 0 | 1.0 | strongest evidence |

- **Weighted EMA:** `m ← clamp(m + α·w·(o − m))`. It is recency-weighted, needs O(1) state, and one noisy signal cannot swing it far.
- **Decay toward a prior**, not toward 0: `m_eff = prior + (m − prior)·2^(−Δt/half_life)`. Forgetting returns you to "what we'd assume about you", not "knows nothing". The prior comes from the expertise level. Decay is applied **when read**, so no nightly job rewrites rows.
- **Expertise is inferred** from the mean effective mastery over concepts with at least 3 pieces of evidence, and only once there are at least 3 such concepts. Too little evidence means no change.
- **SM-2 scheduling** (the SuperMemo/Anki algorithm), driven only by genuine recall signals (quiz, understood, confused). A first exposure schedules a review in one day.
- **Locks.** An explicit value (profile field or concept mastery) is never decayed or inferred over until the user unlocks it. That meets the plan's "explicit override priority".

### Concepts: a curated vocabulary
Mastery needs **stable identities**: "LoRA", "low-rank adaptation" and "LoRA fine-tuning" must update one row. A vocabulary of about 80 ML concepts with aliases does that for free and deterministically. Longest alias wins, and matches must sit on word boundaries.

### Pipeline (`manager.py`, `store.py`)
- `submit()` is `put_nowait` on a bounded queue. A full queue, or a user over `MEMORY_MAX_UPDATES_PER_MINUTE`, **drops** the signal (logged and counted). It never blocks the request.
- One worker applies events. Each event runs in **one transaction**: insert the event id (the idempotency key, `ON CONFLICT DO NOTHING`), compute from a snapshot, upsert mastery `WHERE NOT locked`, patch the profile `WHERE version = :seen` (an atomic `profile || patch`), write audit rows, commit.
- A version mismatch raises `ConcurrentUpdate`, and the event is recomputed at once on a fresh snapshot (optimistic concurrency: no locks held while computing). Other errors are retried with exponential backoff and jitter, up to `MEMORY_MAX_RETRIES`.
- **Audit:** every change to a user-facing field and every mastery change is recorded with old and new values, its source (`explicit` or `inferred`) and the event id. `inferred_interests` is derived and not audited.

### Account data
- Conversations belong to users. A foreign conversation id is a **404** everywhere (read, continue, list).
- The GDPR endpoints are `GET /me/export` (everything, as JSON) and `DELETE /me`. Deletion requires the password again, so a stolen token alone cannot erase an account, and cascades through every table.

## Alternatives considered

| Option | Why not (now) |
|---|---|
| Bayesian Knowledge Tracing / IRT | Principled, but it needs calibrated per-concept parameters we have no data to fit. EMA plus decay is explainable and tunable. Revisit with Phase 5 quiz data. |
| LLM concept extraction | Unstable spellings (duplicate rows), costs quota on every turn, and isn't deterministic in tests. |
| Celery / RQ worker | A broker plus a second process: impossible on a one-container free tier (ADR 0006). |
| FastAPI `BackgroundTasks` | Runs after the response, but there's no queue bound, no per-user cap, no retry policy, and no way to drain on shutdown. |
| Transactional outbox (events written in the request transaction) | Durable across crashes. It's the upgrade path (below), but it puts a DB write on every request's critical path, which the plan's "never block" rules out for soft signals. |
| Row locks (`SELECT … FOR UPDATE`) instead of versions | They hold locks while Python computes. Contention is rare (one worker per process), so optimistic is cheaper. |
| Monthly partitioning of `profile_audit` | The plan suggests it. At free-tier volume (thousands of rows) it adds partition-maintenance jobs for no gain. Index `(user_id, created_at)` now, and partition when the table passes about 10M rows or retention needs a cheap `DROP PARTITION`. |

## Revisit when

- **Several instances or a need for durability:** switch to an outbox table (`user_events` with `processed_at`, written with the turn) drained by `SELECT … FOR UPDATE SKIP LOCKED`. The event-id idempotency already makes redelivery safe.
- Phase 5 quiz data exists: fit the observation weights, or move to BKT.

## Consequences

- Good: every number is explainable ("mastery dropped because you said you were confused, audit #123").
- Good: the model is pure, so it's exhaustively unit-tested. The stores only apply `Changes` atomically.
- Cost: signals queued at the instant of a crash are lost (at-most-once across crashes). They are soft evidence, and the audit shows exactly what was applied.
- Cost: concepts outside the vocabulary are invisible until added.
