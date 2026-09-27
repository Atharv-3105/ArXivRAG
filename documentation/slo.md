# Service level objectives

SLOs are what the alerts in `ops/prometheus/alerts.yml` protect. Each has a metric, a target and a window. The targets come from the plan (§6 Phase 6: "95% citation grounding, < 2 s p95 latency"). "Measured" says whether we have a real number yet.

| SLO | Metric | Target | Window | Measured |
|---|---|---|---|---|
| Availability | share of non-5xx responses (`arxivmind_http_requests_total`) | ≥ 99% | 30 days | no (not deployed) |
| Responsiveness | p95 **time to first token** (`arxivmind_chat_first_token_seconds`) | < 2 s | 7 days | overhead only: p95 26 ms without the LLM; the provider share is unmeasured |
| Grounding | guardrail approval rate (`arxivmind_guardrail_decisions_total`) | ≥ 95% | 7 days | no (needs real models) |
| Degradation | share of degraded answers (`outcome="degraded"`) | ≤ 5% | 7 days | no |
| Freshness | ingestion succeeded in the last 26 h (workflow run status) | 100% of days | 30 days | no |

Why time to first token and not full response time: answers stream. A 12-second answer whose first words appear after 800 ms feels fast, and a 3-second answer that shows nothing for 3 s feels broken. Full duration is still recorded per route.

**Exclusions:** cold starts of a sleeping free-tier instance (see `deployment.md`), requests rejected by limits (429s are the limiter working), and client disconnects.

**Error budget:** at 99% availability that's about 7 hours a month. When it's spent, changes are frozen except fixes, per the runbooks.
