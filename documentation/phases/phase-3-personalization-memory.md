# Phase 3: Personalization, memory and accounts

> Goal (plan §6): user profiles, an async MemoryManager, concept mastery and audit trails. For the SaaS: accounts, per-user limits and GDPR.
> Decisions: [ADR 0008](../adr/0008-learner-model-and-memory-pipeline.md) (learner model and pipeline), [ADR 0006](../adr/0006-single-instance-in-process-state.md) (in-process state, no Redis).

## 1. What happens on a chat turn now

```
POST /api/v1/chat/stream  (Authorization: Bearer <jwt>)
  current_user            JWT verified (HS256, exp, iss), user row loaded (deleted → 401)
  enforce_chat_limits     per-minute sliding window (memory) + daily quota (Postgres count) → 429
  history                 owner-checked: someone else's conversation id → 404
  concepts                vocabulary match on the question: "Explain LoRA" → ["lora"]
  reader context          profile + effective (decayed) mastery of those concepts
  graph                   classify → retrieve → generate, with a "Reader:" block in the prompt
  save turn               user-scoped
  memory.submit(event)    put_nowait → worker → one transaction → mastery, profile, audit
```

The response is never delayed by the profile update: the event is queued, and the worker applies it shortly after.

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register`, `/auth/login` | email + password → bearer token |
| GET | `/me` | account + profile + locked fields |
| PATCH | `/me/profile` | explicit expertise, math depth, goals and interests (locks them); `unlock: [...]` |
| GET | `/me/mastery` | concepts with effective and stored mastery, evidence, lock, next review |
| PUT | `/me/mastery/{concept}` | explicit value (locks it) or `null` (unlock) |
| GET | `/me/reviews` | concepts due for spaced-repetition review |
| GET | `/me/audit` | every profile or mastery change and its source |
| GET | `/me/export` | GDPR export (account, profile, mastery, events, audit, conversations) |
| DELETE | `/me` | GDPR erasure (password required) |
| GET | `/conversations`, `/conversations/{id}` | the user's history |
| POST | `/feedback` | `understood` / `confused` (move mastery), `helpful` / `not_helpful` (answer rating) → 202 |

## 3. Components and the reasoning behind them

### Auth (`src/auth/`)
- **bcrypt** with configurable cost. Passwords over 72 bytes are *rejected*, because bcrypt would silently ignore the rest.
- **Login does not leak which emails exist:** the same error for an unknown email and a wrong password, and the same timing, because an unknown email is still checked against a dummy hash.
- **JWT:** the algorithm is pinned server-side (`algorithms=[...]`, so `alg=none` and algorithm-swap attacks fail). `exp`, `iat` and `iss` are required. The user row is loaded per request, so deleting an account kills its tokens immediately without a revocation list.
- Emails are normalised to lower case, and uniqueness is enforced by the database (`ON CONFLICT`), which avoids a check-then-insert race.

### Limits (`src/api/limits.py`)
- The per-minute limit is a sliding window in memory. It never allows 2× across a boundary, and losing it on restart is harmless.
- The daily quota is counted from Postgres, so a free-host restart cannot reset it. The count **fails open** if the DB is down, and the LLM router's budgets remain as a backstop.
- Admins are exempt. Errors are 429 with `Retry-After`: seconds until the window frees, or until 00:00 UTC for the quota.

### Learner model (`src/memory/model.py`)
EMA on weighted observations, decay toward an expertise prior with a half-life (applied when read), SM-2 scheduling, expertise inferred only from enough evidence, and explicit locks. The maths is all in ADR 0008, and it's all pure functions.

### Pipeline (`src/memory/manager.py`, `store.py`)
A bounded queue with non-blocking submit and a per-user cap. One transaction per event: idempotency insert, guarded mastery upsert, version-checked JSONB patch, audit rows. `ConcurrentUpdate` means an immediate recompute; other errors get backoff with jitter.

### Prompt personalisation (`src/agents/prompts.py`)
A `Reader:` block covers expertise level (how to pitch), math depth, up to three goals, concepts they know (don't re-explain) and concepts they're shaky on (explain carefully). `PROMPT_VERSION` moved to `chat-v2`, so answers before and after are distinguishable in logs.

## 4. Failure modes

| Failure | Behaviour |
|---|---|
| Postgres down during chat | profile → defaults, history → empty, quota → fail open, turn not saved. **The answer is still produced.** |
| Memory worker DB error | retried with backoff, then counted as `failed` and logged. The request never saw it. |
| Queue full or user over cap | signal dropped, counted in `stats`. |
| Same event delivered twice | the second is a no-op (event-id conflict). |
| User edits profile while the worker computes | `ConcurrentUpdate` → recomputed on a fresh snapshot. |
| User locks a concept while the worker writes it | the upsert's `WHERE NOT locked` keeps the user's value. |
| Token for a deleted account | 401 `INVALID_TOKEN`. |

## 5. Acceptance criteria (plan §6, Phase 3)

| Criterion | Status |
|---|---|
| Profile updates asynchronously after response | Done: `test_chat_turn_updates_mastery_asynchronously` (the API returns, then the worker applies). |
| Audit log shows field changes | Done: explicit and inferred, with old/new values and event id. |
| Explicit override pauses inference | Done: profile-field and concept locks, tested at model, store and API level. |
| Mastery scores clamp to [0, 1] | Done: `clamp` in the model, plus a DB `CHECK` constraint. |
| Atomic JSONB patches, optimistic concurrency | Done: `profile \|\| patch WHERE version = :seen`, SQL asserted in tests. |
| Async queue with retries | Done. |
| Audit partitioning | **Deferred on purpose** (ADR 0008: index now, partition past about 10M rows). |
| GDPR export/delete | Done. |
| Live run on Postgres | **Pending**: the Docker engine is unavailable. SQL semantics are asserted by compiling statements against the Postgres dialect. |

## 6. Interview questions (with the answer you should be able to give)

1. **"How do you update a user profile without slowing the response?"** A bounded in-process queue with `put_nowait`: never await it in the request, drop when full, and have one worker apply events in transactions. Know the trade-off (at-most-once across crashes) and the upgrade (an outbox with `SKIP LOCKED`).
2. **"How do you make the update idempotent?"** The event id is the primary key of `user_events`, inserted first in the same transaction as the effects. A redelivery conflicts, and the whole event is skipped.
3. **"Optimistic or pessimistic concurrency here, and why?"** Optimistic: `UPDATE … WHERE version = :seen`, and zero rows means recompute. Contention is rare, and no locks are held while Python computes. Pessimistic (`FOR UPDATE`) suits hot rows with frequent conflicts.
4. **"Why a JSONB `||` patch rather than read-modify-write in Python?"** The merge happens in one statement in the database, so two writers touching different keys can't clobber each other's keys. The version check covers same-key races.
5. **"How does your mastery model work, and why not BKT?"** Weighted EMA of observations, decay toward a prior with a half-life, SM-2 for scheduling. BKT or IRT need fitted parameters we have no data for. EMA is explainable, cheap and tunable, and it's the right v1.
6. **"Why decay at read time?"** No cron job, no write amplification, and it's always exact for "now". Store the value and its timestamp, and compute `m_eff` on read.
7. **"How do explicit overrides interact with inference?"** A lock per field or concept. Inference skips locked items. The DB upsert is guarded (`WHERE NOT locked`) against a race, and unlocking hands the field back.
8. **"How do you prevent user enumeration at login?"** Identical error and identical timing (a dummy bcrypt check for unknown emails). Rate-limit the endpoint in production.
9. **"JWT pitfalls?"** Pin the algorithm server-side (reject `none` and algorithm swaps), require `exp`, keep the payload minimal (no PII), and know revocation: load the user per request or keep short TTLs.
10. **"Why is someone else's conversation a 404 and not a 403?"** A 403 confirms the id exists. A 404 reveals nothing. That's the IDOR (insecure direct object reference) defence.
11. **"How do you implement GDPR erasure?"** `ON DELETE CASCADE` from `users` across every owned table, re-authentication before deletion, and an export endpoint for access and portability. Logs hold ids, not content.
12. **"Rate limiting: where is the state and why?"** The per-minute window is in memory (short-lived, harmless to lose). The daily quota is in Postgres (must survive restarts). With N instances, move both to Redis (ADR 0006).
13. **"Why a concept vocabulary instead of LLM extraction?"** Stable identities for the mastery rows, zero cost and deterministic tests. The cost is coverage, which is the trade-off to name.
