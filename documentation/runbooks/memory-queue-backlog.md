# Memory queue backlog

**Symptoms:** `MemoryQueueBacklog` fires (`arxivmind_memory_queue_depth` > 500 for 10 minutes). Profiles and mastery update late, and past `MEMORY_QUEUE_SIZE` new signals are **dropped** (`arxivmind_memory_events{kind="dropped_full"}`).

**Diagnose**
- `arxivmind_memory_events`:
  - rising `retried`/`failed` means the worker can't write (usually Postgres, see [database-down](database-down.md));
  - a flat `applied` with a growing queue means the worker is stuck or slow.
- Logs: `Memory event failed`, `Memory worker error`.

**Fix**
- Database trouble: fix Postgres first. Queued events apply once it's back (retries with backoff), and events past the retry limit are logged and dropped. They're soft signals (ADR 0008).
- Worker stuck: restart the instance. Queued events are lost on restart, which is at-most-once by design; the audit log shows exactly what was applied.
- Sustained load beyond one worker: the upgrade path is the outbox table plus `SKIP LOCKED` workers (ADR 0008 "Revisit when").

**Verify:** queue depth returns to near 0, and `applied` grows with chat traffic.
