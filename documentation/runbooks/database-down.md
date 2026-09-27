# Postgres unavailable or slow

**Symptoms:**
- `/health/ready` returns 503 with `database: error|timeout`, and the host stops routing traffic to the instance;
- `HighErrorRate` (auth needs the users table) or `GraphRetrievalFailing`;
- answers still arrive but without history or personalisation (these degrade by design).

**Diagnose**
1. `curl /health/ready`: the probe reports the exception *type* (never connection strings).
2. Logs: search for `Could not load history`, `Could not load profile` or `Graph retrieval failed` around the alert time.
3. Provider console: is the free instance paused or over its storage or compute quota? Free Postgres tiers suspend idle databases, and the first connection wakes them (seconds).

**Fix**

| Cause | Action |
|---|---|
| Suspended free instance | usually self-heals on the next connection. If not, resume it in the console |
| Storage quota full | purge old `profile_audit` / `user_events` rows (after exporting), or upgrade |
| Credentials rotated | update `DATABASE_URL` in the host and the Actions secrets, then redeploy |
| Slow queries | the graph queries have `statement_timeout`. Check `pg_stat_statements` for anything else |

**What keeps working:** chat answers (without history), because Qdrant and the LLM don't need Postgres. Registration and login don't.

**Verify:** `/health/ready` is 200, the 5xx rate returns to baseline, and graph status is `ok` again.
