# Deploy and rollback

**Deploy**
1. Merge to `main`, then wait for CI to go green (tests, offline eval gate, secret scan, smoke test). The image `ghcr.io/<owner>/<repo>:<sha>` is pushed.
2. Approve the `production` environment deployment (required reviewers).
3. Point the service at `:<sha>`. The host waits for `/health/ready` = 200 before switching traffic.
4. **Verify** (10 minutes):
   - `curl /health/ready` shows `"status": "ready"`;
   - one chat turn works;
   - on the Grafana dashboard, the 5xx rate, first-token p95 and guardrail approval are unchanged.

**Rollback** (any regression you can tie to the deploy)
1. Point the service back at the previous `:<sha>`, which you'll find in the host's deploy history or the CI run list.
2. Verify as above.
3. Schema: nothing to undo. Every migration is additive (`ADD COLUMN IF NOT EXISTS`, new tables), so the older image ignores the new columns.
4. Open an issue with the trace ids of failing requests, then fix forward.

**Never:** deploy `:latest` directly (you couldn't tell which build you're running), or roll back the database.
