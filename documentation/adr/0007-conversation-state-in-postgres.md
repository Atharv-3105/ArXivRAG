# ADR 0007: Conversation history in our own Postgres tables, not the LangGraph checkpointer

- **Status:** Accepted (2026-09-26)
- **Code:** `src/storage/conversations.py`, `schema.sql` (`conversations`, `messages`), `src/api/routes/chat.py`
- **Tests:** `tests/test_chat_api.py` (history across turns, history outage)

(ADR 0006 is reserved for shared state across multiple instances, rate limits and LLM budgets, written with Phase 3.)

## Context

The plan asks for "LangGraph state persists across turns" and "state checkpointing". LangGraph's answer is a checkpointer, which after every node saves the full graph state as an opaque, serialized blob keyed by `thread_id`.

What the product actually needs to persist between turns is the chat history (the user and assistant messages, with citations). It also needs to:

- list and show past conversations in the UI,
- export and delete a user's data (GDPR, Phase 3),
- sample real questions for the eval set (Phase 6),
- survive restarts, since free hosts sleep and restart often.

## Decision

1. **The graph is stateless per turn.** Each request loads the last `CHAT_HISTORY_MESSAGES` messages, runs classify → retrieve → generate, and saves the turn (user plus assistant message, intent, citations, provider, trace id) in one transaction.
2. **Plain relational tables:** `conversations(id, title, …)` and `messages(conversation_id, role, content, intent, citations JSONB, provider, trace_id, …)`, indexed on `(conversation_id, id)`.
3. **Best effort.** If history cannot be loaded, the answer is generated without it. If a turn cannot be saved, it is logged. A database outage never blocks an answer. A turn whose generation *failed* is not saved.
4. **The store is a protocol** (`ConversationStore`). Postgres runs in production, an in-memory store in tests.

## Alternatives considered

| Option | Why not (now) |
|---|---|
| LangGraph `PostgresSaver` | It writes a checkpoint per node per turn, as opaque blobs that can't be queried for the UI, export or eval. It also needs a second Postgres driver (psycopg 3) beside asyncpg. Its main benefits, resuming a long or interrupted multi-step run and human-in-the-loop pauses, don't apply to a three-node, single-pass graph. |
| `MemorySaver` (in-process) | Lost on every restart, and free hosts restart constantly. |
| Redis for sessions | Another service to run, and history must be durable anyway. |

## Revisit when

- The graph gets long-running or interruptible steps (tool loops, human approval): then add a checkpointer *for runs*, and keep these tables *for history*.

## Consequences

- Good: history is ordinary data: it can be queried, exported, deleted and joined with users (Phase 3).
- Good: each turn's trace id is stored, so a bad answer in the UI links straight to its logs.
- Cost: no mid-run resume. If the process dies mid-answer, the user retries the turn, which is acceptable for a single-pass graph.
