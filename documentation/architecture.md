# ArXivMind — Architecture (as built)

The plan in `Docs/arxiv_plan.md` is the target. This file describes **what exists**, and it lists every deliberate deviation with a link to its ADR. The constraint behind most deviations: **the product must run on free tiers.**

## Layers

```
Clients          Web UI · CLI                                          (Phase 2+)
                     │ HTTPS, SSE
API              FastAPI · pure-ASGI trace-id middleware · typed errors → HTTP
                     │
Agentic core     LangGraph: router → retrieve → specialist → guardrail  (Phase 2+)
                 LLM router: N free providers, failover, budgets, breakers   [ADR 0004]
                     │
Data             Postgres 16 ── source of truth: papers, chunks, equations,
                                references, citation graph, runs, DLQ      [ADR 0002, 0003]
                 Qdrant ─────── derived vector index, rebuildable from Postgres
Background       python -m src.pipeline.runner (cron) · reindex (reconciler)  [ADR 0001]
```

## Stores and ownership

| Data | Store | Written by | Rebuildable? |
|---|---|---|---|
| Paper metadata, parse provenance, `index_status` | Postgres `papers` | pipeline | no (source of truth) |
| Chunk text + section path | Postgres `paper_chunks` | pipeline | no |
| Equations (LaTeX), references | Postgres | pipeline | no |
| Citation edges | Postgres `paper_citations` | pipeline, in the paper's own transaction | yes, from references |
| Dense vectors + payload | Qdrant | pipeline / `reindex` | **yes, from Postgres** |
| Failed papers | Postgres `failed_papers` (DLQ) | pipeline | n/a |
| Run audit + metrics | Postgres `ingestion_runs` | pipeline | n/a |

## Deviations from the plan

| Plan says | Built | Why | ADR |
|---|---|---|---|
| Neo4j for the citation graph | Postgres edge table plus recursive CTEs | One free store, atomic with paper writes, and our queries are shallow | [0003](adr/0003-citation-graph-in-postgres.md) |
| `unstructured` + pdfplumber + Nougat | arXiv HTML (real LaTeX) → PyMuPDF fallback | Exact math without a GPU, measured at 99.54% of formulas verbatim | [0001](adr/0001-html-first-ingestion.md) |
| BGE-large embeddings | bge-small (384-d), configurable | CPU-only free host, and a model swap is a `reindex --all` | [phase 1](phases/phase-1-ingestion.md) |
| Configurable Ollama / OpenAI / Anthropic | A router over free providers (Groq, Gemini, Cerebras, OpenRouter, Ollama) plus optional Anthropic and an offline mock | Free-tier quotas are small, so one provider means downtime | [0004](adr/0004-llm-router.md) |

## Cross-cutting rules (enforced in code review)

- **Config-driven.** Model names, URLs, thresholds and limits live in `src/core/config.py` (pydantic-settings), and startup fails on invalid config.
- **Every external call has a timeout, a retry policy that only retries transient errors, and a fallback.**
- **Idempotency everywhere:** `ON CONFLICT`, deterministic ids, status flags.
- **Provenance travels with data:** `parse_method` and `math_fidelity` reach the vector payload, so answers can disclose degraded math.
- **Structured logs** (structlog JSON) carry a `trace_id` from the HTTP middleware through to every log line.

## Documents

- Phases: [phase 1 — ingestion](phases/phase-1-ingestion.md)
- ADRs: [0001](adr/0001-html-first-ingestion.md) · [0002](adr/0002-postgres-source-of-truth.md) · [0003](adr/0003-citation-graph-in-postgres.md) · [0004](adr/0004-llm-router.md)
- [Interview questions](interview_questions.md)
