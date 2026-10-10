# ArXivMind — Architecture (as built)

The plan in `Docs/arxiv_plan.md` is the target. This file describes **what exists**, and it lists every deliberate deviation with a link to its ADR. The constraint behind most deviations: **the product must run on free tiers.**

## Layers

```
Clients          Web app: React SPA (Margin design + Atlas map), served by the API [ADR 0014]
                     │ HTTPS, SSE
API              FastAPI · JWT auth · per-user limits · /chat, /chat/stream (SSE), /me, /feedback
                 /metrics (Prometheus) · /health/live · /health/ready                    [ADR 0012]
                 trace-id middleware · typed errors → HTTP                   [ADR 0006]
                     │
Agentic core     LangGraph: route → retrieve (hybrid, RRF, + graph) → specialist   [ADR 0005, 0009, 0010]
                 specialists: tutor (stateful, Socratic) · critic · synthesizer · advisor · general
                 guardrail: input gate (domain, injection, opt-in moderation) + grounding checks [ADR 0011]
                 (specialists: Phase 5, guardrail: Phase 6)
                 LLM router: N free providers, failover, budgets, breakers   [ADR 0004]
                 MemoryManager: async queue → learner model → profile/mastery [ADR 0008]
                     │
Data             Postgres 16 ── source of truth: papers, chunks, equations,
                                references, citation graph + contexts,      [ADR 0002, 0003]
                                influence, runs, DLQ                         [ADR 0009]
                 Qdrant ─────── derived vector index, rebuildable from Postgres
Background       scheduled GitHub Actions ingestion · reindex (reconciler)   [ADR 0001, 0013]
Ops              CI gate (tests + offline eval + secret scan) · GHCR sha images · runbooks [ADR 0013]
```

## Stores and ownership

| Data | Store | Written by | Rebuildable? |
|---|---|---|---|
| Paper metadata, parse provenance, `index_status` | Postgres `papers` | pipeline | no (source of truth) |
| Chunk text + section path | Postgres `paper_chunks` | pipeline | no |
| Equations (LaTeX), references | Postgres | pipeline | no |
| Citation edges | Postgres `paper_citations` | pipeline, in the paper's own transaction | yes, from references |
| Citation contexts (sentence + function) | Postgres `citation_contexts` | pipeline, same transaction | yes, by re-parsing cached HTML |
| Influence (PageRank) | Postgres `paper_influence` | after each run / reindex | yes, recomputed |
| Dense + BM25 vectors + payload | Qdrant | pipeline / `reindex` | **yes, from Postgres** |
| Conversations, messages (per user), tutor session state, prompt versions, checks | Postgres | chat API, best effort | no |
| Users, profiles, concept mastery | Postgres | auth, `/me`, MemoryManager | no |
| Processed events, profile audit | Postgres `user_events`, `profile_audit` | MemoryManager | no |
| Failed papers | Postgres `failed_papers` (DLQ) | pipeline | n/a |
| Run audit + metrics | Postgres `ingestion_runs` | pipeline | n/a |

## Deviations from the plan

| Plan says | Built | Why | ADR |
|---|---|---|---|
| Neo4j for the citation graph | Postgres edge table plus recursive CTEs | One free store, atomic with paper writes, and our queries are shallow | [0003](adr/0003-citation-graph-in-postgres.md) |
| `unstructured` + pdfplumber + Nougat | arXiv HTML (real LaTeX) → PyMuPDF fallback | Exact math without a GPU, measured at 99.54% of formulas verbatim | [0001](adr/0001-html-first-ingestion.md) |
| BGE-large embeddings | all-MiniLM-L6-v2 (384-d) on ONNX (fastembed), configurable; BM25-only fallback if it fails | CPU-only free host with no PyTorch, a model that cannot become a hosting constraint, and a swap is a `reindex --all` | [0005](adr/0005-hybrid-retrieval-rrf.md), [0015](adr/0015-curated-corpus-and-minilm.md) |
| Papers found by search query | A curated, verified list of 236 foundational papers (`corpus/foundational.csv`) | Grounded answers for what learners ask, and a reproducible corpus for evaluation | [0015](adr/0015-curated-corpus-and-minilm.md) |
| LangGraph checkpointing | Stateless graph per turn, history in Postgres tables | Queryable history (UI, GDPR, evals), survives restarts | [0007](adr/0007-conversation-state-in-postgres.md) |
| Redis for sessions, rate limits, hot profiles | Stateless JWT; in-process windows; Postgres reads; daily quota in Postgres | One free instance; no extra service or hop | [0006](adr/0006-single-instance-in-process-state.md) |
| Celery/RQ worker, partitioned audit | In-process asyncio queue + transactional apply; indexed audit | One container; volume too low to partition | [0008](adr/0008-learner-model-and-memory-pipeline.md) |
| Tutor/Critic/Synthesizer/Advisor as agents | Specialist graph nodes, ≤ 1 LLM call per turn; Advisor ranks deterministically | Free-tier quota and latency; predictable tasks | [0010](adr/0010-specialist-agents.md) |
| LangSmith for eval and A/B | Own harness: offline CI gate, live suites, A/B from the messages table | No cloud dependency or keys in CI; gates are deterministic | [0012](adr/0012-evaluation-and-observability.md) |
| Airflow (Phase 2 of orchestration) | Scheduled GitHub Actions workflow | A daily batch needs no scheduler service on a free tier | [0013](adr/0013-deployment-and-operations.md) |
| Configurable Ollama / OpenAI / Anthropic | A router over free providers (Groq, Gemini, Cerebras, OpenRouter, Ollama) plus optional Anthropic and an offline mock | Free-tier quotas are small, so one provider means downtime | [0004](adr/0004-llm-router.md) |

## Cross-cutting rules (enforced in code review)

- **Config-driven.** Model names, URLs, thresholds and limits live in `src/core/config.py` (pydantic-settings), and startup fails on invalid config.
- **Every external call has a timeout, a retry policy that only retries transient errors, and a fallback.**
- **Idempotency everywhere:** `ON CONFLICT`, deterministic ids, status flags.
- **Provenance travels with data:** `parse_method` and `math_fidelity` reach the vector payload, so answers can disclose degraded math.
- **Structured logs** (structlog JSON) carry a `trace_id` from the HTTP middleware through to every log line.

## Documents

- Phases: [phase 1 — ingestion](phases/phase-1-ingestion.md) · [phase 2 — agentic loop](phases/phase-2-agentic-loop.md) · [phase 3 — personalization](phases/phase-3-personalization-memory.md) · [phase 4 — graph retrieval](phases/phase-4-graph-retrieval.md) · [phase 5 — agents & pedagogy](phases/phase-5-agents-pedagogy.md) · [phase 6 — eval & ops](phases/phase-6-eval-ops.md)
- ADRs: [0001](adr/0001-html-first-ingestion.md) · [0002](adr/0002-postgres-source-of-truth.md) · [0003](adr/0003-citation-graph-in-postgres.md) · [0004](adr/0004-llm-router.md) · [0005](adr/0005-hybrid-retrieval-rrf.md) · [0006](adr/0006-single-instance-in-process-state.md) · [0007](adr/0007-conversation-state-in-postgres.md) · [0008](adr/0008-learner-model-and-memory-pipeline.md) · [0009](adr/0009-graph-enhanced-retrieval.md) · [0010](adr/0010-specialist-agents.md) · [0011](adr/0011-guardrail.md) · [0012](adr/0012-evaluation-and-observability.md) · [0013](adr/0013-deployment-and-operations.md) · [0014](adr/0014-web-frontend.md) · [0015](adr/0015-curated-corpus-and-minilm.md)
- [Web app](frontend.md) · [SLOs](slo.md) · [Deployment](deployment.md) · [Runbooks](runbooks/README.md) · [Evaluation](eval/README.md) · [API](api/README.md)
- [Interview questions](interview_questions.md)
