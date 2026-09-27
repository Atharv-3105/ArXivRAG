# ADR 0012: Evaluation in three layers, and Prometheus metrics without cardinality traps

- **Status:** Accepted (2026-09-27). Plan Phase 6 (eval harness, Prometheus/Grafana, alerting, load and chaos testing).
- **Code:** `src/eval/` (offline, live, ab, scoring, seed), `loadtest/` (bench, run), `src/core/metrics.py`, `src/api/routes/health.py`, `ops/`
- **Tests:** `tests/test_ops.py` (metrics, probes, chaos matrix, the offline gate), `tests/test_guardrail.py`

## Context

Every phase left claims open: does BM25 add to dense, does the graph help, do real models follow the tutor's trailer contract, is the query prefix useful, and does the product meet "95% citation grounding, < 2 s p95"? Some can be measured without services or keys. Some need the corpus. Some need real models and real users.

## Decision

### Evaluation in three layers
| Layer | Needs | Runs | Measures |
|---|---|---|---|
| **Offline** (`python -m src.eval.offline`) | nothing | every CI run, **gates the build** | routing, domain gate, tutor action detection, citation function, guardrail decisions, template validity: every rule-based decision, against labelled data |
| **Live** (`python -m src.eval.live retrieval \| answers`) | seeded corpus (`python -m src.eval.seed`), LLM keys | on demand, before and after changes | retrieval ablation (dense / sparse / hybrid / hybrid+graph: recall@5, MRR, nDCG@10); end-to-end turns (time to first token p50/p95, check pass rates, guardrail approval) |
| **Production** (`python -m src.eval.ab`) | the messages table | periodically | per prompt version: guardrail approval, one-question and trailer compliance, grounding |

Rules for the labelled data:
- It is written from realistic phrasing *before* the suites are first run.
- Failures are listed in the report, not hidden.
- Thresholds (`src/eval/data/thresholds.json`) are never lowered to make a change pass.
- When the data informs a fix, the report says so.

### Metrics
- `prometheus_client` with a dedicated registry and the `/metrics` endpoint (optional bearer token).
- **Bounded labels only:** route *templates* (rebuilt from the mounted router's relative path plus prefix), agent names, provider slots, issue codes. There are never user ids, queries or paper ids.
- The headline latency is **time to first token** (`arxivmind_chat_first_token_seconds`), since that's the latency a streaming user feels. Full request duration is kept per route.
- Runtime state (memory queue depth, circuit breakers) is exported by a collector read at scrape time, so the hot path never updates a gauge for it.
- Probes: `/health/live` (process) and `/health/ready` (database required; Qdrant reported but not required, because the product degrades without it).

### Alerting and dashboards
- `ops/prometheus/alerts.yml`: every alert derives from an SLO (`documentation/slo.md`) and names its runbook.
- `ops/grafana/.../arxivmind.json`: 16 panels.
- The whole stack runs locally with `docker compose --profile observability up`.

### Load, latency and chaos
- `loadtest/bench.py` measures **our own overhead** in-process (LLM and stores faked). If overhead alone threatened the 2 s budget, no provider choice could save it.
- `loadtest/run.py` is an HTTP load test for a running deployment (no extra dependencies).
- The chaos matrix (`tests/test_ops.py`) injects Qdrant down, history store down, profile store down, an LLM timeout with failover, and every LLM timing out. It asserts degraded answers or 503 + `Retry-After`, never a 500 or a hang.

## Measured so far (2026-09-27)

| What | Result |
|---|---|
| Offline suites | routing 97.8%, domain gate 0% false refusals / 100% off-topic recall, tutor actions 100%, citation function macro-F1 0.854, guardrail 100%, templates valid. All gates met. |
| App overhead per turn (mocked LLM) | p50 22 ms, p95 26 ms, about 45 turns/s per process |
| At 10 concurrent turns | p95 229 ms, which is queueing in one event loop, not per-request cost |

Fixes found by the eval: a tie between generic question openers and specific cues (routing), and plural concept names ("VAEs") being refused by the domain gate.

**Not yet measured** (need Docker and keys): the retrieval ablation, real-model compliance and time to first token, image size and RSS, and production A/B.

## Alternatives considered

| Option | Why not |
|---|---|
| LangSmith / hosted eval platforms as the eval harness | Cloud dependency and account needed. The plan keeps LangSmith optional. The harness here is plain Python and JSON reports that CI can gate on. |
| LLM-judged answers as the CI gate | Non-deterministic, costs quota, needs keys in CI. Offline rule suites gate; LLM judging belongs in the live layer. |
| Locust / k6 | Extra dependency or binary. An asyncio + httpx script covers streaming time to first token, which needs custom code anyway. |
| OpenTelemetry tracing + collector | Right at scale; a collector is another service on a free tier. Structured logs with trace ids already give per-request traces. Revisit with multiple services. |
| Per-user or per-paper metric labels | Unbounded series count, the classic way to take down Prometheus. Use logs (they have trace ids) for per-request questions. |

## Consequences

- Good: every rule the product relies on has a number, a threshold and a CI gate. The open questions from earlier ADRs each have a named, runnable measurement.
- Honest gap: the *live* numbers, and therefore the plan's acceptance targets (95% grounding, < 2 s p95), are **unverified** until the live suites run against the seeded corpus with real providers.
