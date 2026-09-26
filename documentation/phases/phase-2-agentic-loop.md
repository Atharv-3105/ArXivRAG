# Phase 2: Core agentic loop

> Goal (plan §6): a LangGraph workflow, Router → Retrieve → Generate → Stream, with citations.
> Decisions: [ADR 0005](../adr/0005-hybrid-retrieval-rrf.md) (hybrid retrieval), [ADR 0007](../adr/0007-conversation-state-in-postgres.md) (conversation state), [ADR 0004](../adr/0004-llm-router.md) (LLM router, built earlier).

## 1. Request path

```
POST /api/v1/chat/stream {message, conversation_id?}
  TraceIDMiddleware        trace id bound to every log line, echoed as X-Trace-ID
  validation               400 on bad input (length ≤ MAX_MESSAGE_CHARS, UUID conversation id)
  load history             last CHAT_HISTORY_MESSAGES messages (best effort)
  LangGraph
    classify               rules (free) or LLM (opt-in) → learn|critique|synthesize|recommend|chat
      ├─ smalltalk ────────────────────────────────┐
    retrieve               hybrid search (dense + BM25, RRF), per-paper cap, [n]-numbered sources
    generate               grounded prompt → LLM router stream → tokens → [n] → citations
  save turn                user + assistant message, intent, citations, provider, trace id
  SSE events               meta · intent · sources · provider · token* · done | error
```

Try it (no API key needed, since the offline mock answers extractively from the sources):

```bash
uvicorn src.api.app:app --reload
curl -N -X POST localhost:8000/api/v1/chat/stream -H 'content-type: application/json' -d '{"message":"Explain how LoRA works"}'
```

## 2. Components and the reasoning behind them

### State (`src/agents/state.py`)
`AgentState` is a `TypedDict`: the contract between nodes. Each node returns only the keys it changes. `timings` has a reducer (`operator.or_`), so every node's duration is *merged* in rather than overwritten. That is the LangGraph way to let many nodes write to one key.

### Classify (`src/agents/intent.py`)
- Rules by default: free, deterministic, and about 0 ms. Every LLM call on a free tier spends quota that the answer needs more.
- Confidence is the *margin* between the best and second-best intent: `top / (top + second)`. Below `ROUTER_MIN_CONFIDENCE` it returns `chat`, because a wrong specialist is worse than a general answer (plan §4: "defaults to chat on low confidence").
- The LLM classifier (`ROUTER_USE_LLM=true`) uses JSON mode plus a validator. Invalid JSON counts as a provider failure, so the router fails over. Any failure falls back to the rules.
- Smalltalk ("thanks!") skips retrieval entirely through a **conditional edge**.

### Retrieve (`graph.py`, `vector_db.py`)
- Hybrid search with RRF, one Qdrant call (ADR 0005).
- **Follow-ups:** a message of `FOLLOWUP_MAX_WORDS` words or fewer is searched together with the previous user message, so "and its limitations?" retrieves about the right thing. An LLM "standalone question" rewrite is better but costs a call per turn. The trade-off is named, and the eval will measure it.
- **Diversity cap:** at most `RETRIEVAL_MAX_PER_PAPER` chunks per paper.
- Sources get deep links: `arxiv.org/html/<id>#<anchor>` for HTML-parsed papers, `/abs/<id>` otherwise.

### Generate (`graph.py`, `prompts.py`, `grounding.py`)
- One grounded system prompt with numbered sources and a per-intent task line. `PROMPT_VERSION` is logged with each answer, which is ready for Phase 5's versioned templates and A/B tests.
- Degraded-math sources are labelled `(math: degraded)`, and the model is told not to quote their formulas as exact. This is ADR 0001's provenance paying off.
- `learn` uses the router's `tutor` provider order (strongest free reasoning model first), everything else `default`.
- After generation, `[n]` markers are resolved to sources. Markers that point at nothing are recorded as `ungrounded` and logged: the cheap, deterministic half of hallucination detection. The Guardrail (Phase 6) adds the semantic half.

### Failure policy: never block, never invent

| Failure | Behaviour |
|---|---|
| Qdrant down or search error | A fixed "search unavailable" answer, **no LLM call**, `degraded: retrieval_unavailable`. |
| Nothing retrieved (not `chat`) | A fixed "not in the indexed papers" answer, no LLM call, `degraded: no_sources`. |
| Every LLM provider unavailable | `/chat` → 503 + `Retry-After`. `/chat/stream` → an `error` event with `retry_after`. The turn is not saved. |
| Provider dies mid-stream | `LLMStreamInterruptedError` becomes an `error` event. There's no silent switch to another model mid-sentence (ADR 0004). |
| Postgres down (history) | The answer is still produced without history. The failure is logged. |
| Qdrant down at startup | The API starts anyway and answers degrade. A *wrong collection layout* stops startup, because that never heals. |

### Streaming (`routes/chat.py`)
- The **same compiled graph** serves both endpoints. Nodes push events through LangGraph's custom stream writer, which is a no-op under `ainvoke`.
- **SSE over WebSockets:** the traffic is one-way, it's plain HTTP through any proxy or CDN, browsers reconnect natively, and the server holds no connection state.
- `Cache-Control: no-transform` and `X-Accel-Buffering: no` stop proxies from buffering the stream into one late chunk.
- `sources` is sent **before** the first token, so the UI can show what the answer is based on while it streams.

### Errors (`src/api/errors.py`)
Every failure has one shape: `{"error": {"code", "message", "trace_id"}}`. `ArxivMindError.status_code` drives the HTTP status. `Retry-After` comes from the exception. Validation errors are 400 with per-field details. Unexpected errors are 500 with the trace id and no internals.

### Observability
Each node logs `node`, `duration_ms` and output keys, plus the request's `trace_id` from contextvars (plan §8). LangSmith is opt-in (`LANGSMITH_TRACING=true`): retrieve and generate are `@traceable` spans, and LangGraph traces the graph.

## 3. Acceptance criteria (plan §6, Phase 2)

| Criterion | Status |
|---|---|
| Query returns streamed response with citations | Done: tested at the graph and API level (`test_chat_graph.py`, `test_chat_api.py`). |
| State persists across turns | Done: history in Postgres (ADR 0007). Tested through the API with the in-memory store. |
| Trace shows node execution times | Done: `timings` in the `done` event and structured logs. LangSmith is optional. |
| Fallback on LLM timeout | Done: router failover and breakers (ADR 0004), then 503/`error` with `Retry-After`. |
| Mock LLM first, real provider via config | Done: the offline mock is the default with no keys. Setting any provider key enables it. |
| Live run against Qdrant and Postgres | **Pending**: the Docker engine was not responding. Steps below. |

Live verification, once Docker is up:

```bash
docker compose up -d db qdrant
python -m src.pipeline.reindex --all      # builds arxiv_papers_v2 (hybrid) from Postgres
uvicorn src.api.app:app
curl -N -X POST localhost:8000/api/v1/chat/stream -H 'content-type: application/json' -d '{"message":"Explain LoRA"}'
```

## 4. Interview questions (with the answer you should be able to give)

1. **"Why hybrid search? How does RRF work?"** Dense embeddings match meaning, and BM25 matches exact tokens like model names and acronyms. RRF scores a document Σ 1/(k + rank) across the lists, so there is no score calibration between cosine and BM25. It runs in one Qdrant request with prefetch.
2. **"Why is IDF applied by Qdrant rather than at indexing?"** IDF depends on the whole corpus. With `Modifier.IDF`, Qdrant keeps the statistics current, and stored document vectors never go stale as papers are added.
3. **"How do you prevent hallucinated citations?"** The model sees numbered sources only. Every `[n]` is resolved back to a retrieved chunk, and unknown numbers are flagged. With no sources, the answer is fixed and no LLM is called. The semantic check (does the source *support* the claim) is the Guardrail's job.
4. **"Why rules for intent classification?"** They cost zero quota, add no latency and are deterministic. The margin gives a confidence, low confidence falls back to chat, and there's an LLM classifier behind a flag that falls back to the rules.
5. **"SSE or WebSockets for token streaming?"** SSE: the traffic is one-way, it works over plain HTTP and proxies, reconnects automatically and needs no server-side connection state. Mention the buffering headers.
6. **"How does the same graph serve streaming and non-streaming?"** LangGraph's custom stream writer is a no-op under `ainvoke`. Under `astream(stream_mode=["custom", "updates"])`, the API forwards custom events and reads node updates.
7. **"What happens mid-stream if the LLM fails?"** Before the first token, the router fails over. After it, you surface an error rather than continuing on another model, since mixing two models' output in one answer is worse than a clear retry.
8. **"Why not LangGraph's checkpointer for memory?"** Checkpoint blobs can't be queried, and we need history for the UI, GDPR and evals. A single-pass graph needs no mid-run resume. Add a checkpointer when there are interruptible multi-step runs (ADR 0007).
9. **"How do you handle follow-up questions in retrieval?"** Short follow-ups are concatenated with the previous user turn. The alternative is an LLM query rewrite: better quality at one extra call. Know the trade-off and how you'd measure it (retrieval recall on a follow-up eval set).
10. **"Why ONNX embeddings instead of PyTorch?"** Same model weights without PyTorch, the heaviest dependency, so it can fit a free 512 MB host. Be ready to quote the measured image size and RSS (a Phase 6 task) rather than folklore.
11. **"What does the per-paper cap solve?"** Rank concentration: one long, relevant paper can fill all top-k slots, which hurts comparison questions. It's a cheap stand-in for MMR (maximal marginal relevance).
12. **"How would you debug a bad answer a user reports?"** They quote the trace id (it is in the error body and the `X-Trace-ID` header, and stored on the message). From it you get every log line of that request: intent and confidence, retrieval query, sources, provider, per-node timings and prompt version.
