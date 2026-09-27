# Phase 6: Guardrail, evaluation, observability, CI/CD and operations

> Goal (plan §6): production readiness: a Guardrail, an eval harness, Prometheus/Grafana/alerting, GitHub Actions, runbooks and ADRs, load and chaos testing.
> Decisions: [ADR 0011](../adr/0011-guardrail.md) (guardrail), [ADR 0012](../adr/0012-evaluation-and-observability.md) (eval + metrics), [ADR 0013](../adr/0013-deployment-and-operations.md) (deploy + ops).
> Also: [slo.md](../slo.md), [deployment.md](../deployment.md), [runbooks/](../runbooks/README.md), [eval/](../eval/README.md), [api/](../api/README.md).

## 1. The request path now

```
route: moderation (opt-in) → intent + tutor commands → domain gate (free-form only)
  → retrieve → specialist → guardrail (grounding checks; visible notice if not approved) → END
```

Every turn records: the agent, the prompt version, the output checks, the guardrail decision and input flags (stored with the message), and metrics (turn outcome, first token, guardrail, graph status, LLM calls).

## 2. What was built

| Area | Pieces |
|---|---|
| Guardrail | domain gate (no LLM call for off-topic), prompt-injection flag, optional LLM moderation (fails open), post-answer checks: ungrounded, uncited, weak support, degraded or malformed math |
| Evaluation | offline suites as a CI gate (routing, domain, tutor, citation function, guardrail, templates); live retrieval ablation and answer suite; A/B report from stored messages; seed corpus of 29 papers by id |
| Observability | `/metrics` (bounded labels, route templates), `/health/live`, `/health/ready`, runtime collector (queue depth, breakers), 8 SLO-derived alerts, a 16-panel Grafana dashboard, compose profiles `api` and `observability` |
| Resilience | chaos matrix (Qdrant, history store and profile store down; LLM timeout with failover; every LLM timing out → 503), in-process benchmark, HTTP load test |
| CI/CD | offline eval gate, gitleaks over full history, image size report, smoke test asserting *no torch* in the image, push to GHCR with sha tags on `main`; scheduled ingestion workflow |
| Ops | JWT key rotation window (`SECRET_KEY_PREVIOUS`), metrics token, Render blueprint, 9 runbooks, SLOs, OpenAPI export |

## 3. Numbers measured in this phase

| Measurement | Result |
|---|---|
| Offline gate | routing 97.8%, domain gate 0% false refusals / 100% off-topic recall, tutor actions 100%, citation function macro-F1 0.854, guardrail 100%, templates valid |
| App overhead per turn (LLM faked) | p50 22 ms, p95 26 ms (1.3% of the 2 s budget), about 45 turns/s per process |
| Defects the eval found and fixed | routing tie between generic and specific cues (82% → 98%); plural concepts refused ("VAEs"); acknowledgements and statistics refused |

## 4. Acceptance criteria (plan §6, Phase 6)

| Criterion | Status |
|---|---|
| 95% citation grounding | **Measured by the guardrail on every answer, but not yet on real models.** Needs `src.eval.live answers` with keys. |
| < 2 s p95 latency | **Our share is measured:** 26 ms p95 overhead. The provider share (time to first token) is unmeasured; the metric and alert are in place. |
| Eval suite passes | **Yes:** offline gate green, enforced in CI. |
| CI green | **Locally equivalent:** ruff, format, mypy, 495 tests and the offline gate all pass. The workflows haven't run on GitHub (no remote). |
| Runbooks complete | 9 runbooks, and every alert names one. |
| Zero secrets in code | gitleaks in CI over full history; secrets only via env and secret stores; `render.yaml` uses `sync: false`. |
| SLOs, alert thresholds | `slo.md` plus `alerts.yml`. |
| Chaos testing | the matrix in `tests/test_ops.py`. |
| Blue-green, rollback | health-gated platform deploys, sha-tagged images, additive-only schema. **Not exercised:** nothing is deployed. |
| Load testing | `loadtest/run.py` is ready. **Not run** against a deployment. The in-process benchmark was run. |

## 5. Interview questions (with the answer you should be able to give)

1. **"How do you guardrail a *streamed* answer?"** You can't withhold what's already shown. Gate cheaply *before* generation (domain, moderation). Check *after* (grounding), then flag visibly and record, rather than silently rewrite. Buffering would break time to first token.
2. **"How do you check that an answer is grounded without a second LLM call?"** Deterministic layers: every `[n]` must exist, the share of uncited substantive sentences, lexical support between each cited sentence and its source, and formula provenance. Keep LLM judging for offline eval, and know where lexical checks fail (paraphrase-level contradiction).
3. **"Why a domain gate, if retrieval already finds nothing for off-topic questions?"** Vector search always returns *something*. Without a gate, the LLM improvises from the nearest ML chunks, and quota is spent. Tune it for false refusals first, since refusing a real user is the worse error.
4. **"How do you evaluate an LLM product in CI without keys or flakiness?"** Evaluate the deterministic decisions (routing, gates, parsers, guardrail) against labelled data, gate on thresholds, and keep model-dependent evals in a separate live layer.
5. **"Your eval found bugs you then fixed. Aren't the numbers now biased?"** Yes, optimistically. Fix causes, not cases; say which data informed which fix; and hold out a set written by someone else.
6. **"What's your latency SLO, and why time to first token?"** Streaming answers are judged by when words appear. Split latency into our overhead (measured: 26 ms p95) and the provider's (monitored per provider), because they need different fixes.
7. **"How do you avoid killing Prometheus?"** Bounded labels only: route templates, never raw paths, user ids or queries. Runtime state goes through a scrape-time collector.
8. **"Walk me through a rollback."** Point the service at the previous immutable sha image tag. It's safe because migrations are additive only (expand/contract). Health-gated deploys mean a broken image never takes traffic.
9. **"How do you rotate a JWT secret without logging everyone out?"** A verification window: sign with the new key, still accept the previous one until the token TTL passes, then drop it. On a leak, skip the window.
10. **"What did chaos testing prove?"** Each dependency failure maps to a defined degradation (no history, no personalisation, "search unavailable", provider failover, 503 + `Retry-After`) and never to a 500 or a hang, and each is asserted in a test.
