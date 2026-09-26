# Interview questions — index

Each phase document ends with the questions it prepares you for, each with the answer you should be able to give. This page is the index, plus the questions that cut across phases.

| Topic | Where |
|---|---|
| Ingestion, idempotency, dual-store consistency, chunking, embeddings, citation extraction | [Phase 1](phases/phase-1-ingestion.md#5-interview-questions-with-the-answer-you-should-be-able-to-give) |
| Hybrid retrieval, RRF, LangGraph, SSE streaming, grounding, conversation state | [Phase 2](phases/phase-2-agentic-loop.md#4-interview-questions-with-the-answer-you-should-be-able-to-give) |
| Async memory, learner model (EMA/decay/SM-2), idempotency, optimistic concurrency, JWT, GDPR | [Phase 3](phases/phase-3-personalization-memory.md#6-interview-questions-with-the-answer-you-should-be-able-to-give) |
| Graph + vector fusion, PageRank, recursive CTEs, citation function, re-ranking | [Phase 4](phases/phase-4-graph-retrieval.md#6-interview-questions-with-the-answer-you-should-be-able-to-give) |
| LLM routing, failover, rate limiting, circuit breakers | [ADR 0004](adr/0004-llm-router.md) (Decision and Alternatives sections) |

## Cross-cutting

1. **"Walk me through the architecture."** Start from the data (Postgres as source of truth, Qdrant as derived index), then the pipeline that fills it, then the request path. Name one trade-off per layer and its ADR.
2. **"Where did you deviate from the original design, and how did you decide?"** Every deviation has an ADR with context, alternatives and a *revisit-when* trigger. Examples: Neo4j → Postgres, bge-large → bge-small, a single LLM provider → a router.
3. **"What happens when X is down?"** arXiv down: the run fails and is recorded, and nothing half-written is left behind. Qdrant down: papers stay `pending` and are repaired later. An LLM provider down: its breaker opens and traffic fails over. Every provider down: a 503 with `Retry-After`, and the request never hangs.
4. **"How do you test code that talks to five external services?"** Pure functions get unit tests. Stores are faked with stateful in-memory "worlds" (`tests/test_runner.py`), so crash and retry sequences are testable. SQL is asserted by compiling statements against the Postgres dialect. The LLM router is tested with a fake clock. Live checks are run by hand and recorded in the phase docs.
5. **"How is this free to run?"** Free LLM tiers behind a router, a CPU embedding model, one Postgres and one Qdrant (both have free managed tiers), and no always-on worker: ingestion is a cron job.
