# ADR 0004 — Multi-provider LLM router with failover, rate budgets and circuit breakers

- **Status:** Accepted (2026-09-26)
- **Code:** `src/llm/` (`router.py`, `registry.py`, adapters), tests in `tests/test_llm_router.py`, `tests/test_llm_adapters.py`

## Context

ArXivMind must run on free tiers. Every free LLM tier has a small, provider-specific quota (requests per minute, tokens per minute, sometimes requests per day), and free models are deprecated or overloaded without notice. A single provider therefore means: the product stops whenever one quota is hit or one model disappears.

The plan (§5) also requires "Configurable: Ollama / OpenAI / Anthropic" and "local-first privacy mode", and §8 forbids hardcoded model names.

## Decision

A router in front of N **provider slots** (one model on one provider = one slot):

1. **Per-task preference order** (`LLM_TASK_ORDERS`): the cheap intent router prefers the fastest provider; tutoring prefers the strongest free reasoning model. Entries may be prefixes (`openrouter` matches every `openrouter:<model>`). The offline mock is always last.
2. **Failover**: a request tries each available slot at most once, in order.
3. **Error classification from status codes / exception types, never string matching** (`ErrorKind`): `RATE_LIMITED` (429), `TRANSIENT` (5xx, timeout, network), `AUTH` (401/403), `BAD_REQUEST`, `EMPTY`, `INVALID_OUTPUT` (caller validation failed), `REFUSAL`.
4. **Rate budgets**: a sliding 60-second window of requests and tokens, reserved *before* the call and reconciled with real usage after. Slots can share a **rate group** because OpenRouter's free limit is per account, not per model.
5. **429 cooldown** from `Retry-After` (or `x-ratelimit-reset-*`), applied to the whole rate group.
6. **Circuit breaker** per slot: CLOSED → (N consecutive penalised failures) → OPEN → (recovery time) → HALF_OPEN (exactly one probe) → CLOSED on success, or OPEN with doubled recovery (capped). `AUTH` opens immediately. `RATE_LIMITED` and `REFUSAL` never trip it.
7. **Streaming fails over only before the first token.** After text has reached the user, a failure surfaces as `LLMStreamInterruptedError` rather than silently continuing on another model.
8. **Fail fast**: when nothing is available the router raises `LLMUnavailableError` with a `retry_after` hint (the API maps it to HTTP 503 + `Retry-After`) instead of sleeping in the request path.
9. **Provider SDKs are avoided where a protocol suffices**: Groq, Gemini, Cerebras, OpenRouter and Ollama all speak the OpenAI-compatible chat-completions protocol, so one `httpx` adapter covers them. Claude uses the official Anthropic SDK (with SDK retries disabled — the router owns retries).
10. **Configuration**: a provider is enabled by setting its key. `LLM_PROVIDERS_JSON` replaces the defaults; keys are referenced by env-var name, never inlined.

## Alternatives considered

| Option | Why not |
|---|---|
| One provider (e.g. only Groq) | One quota or one model deprecation takes the product down. |
| LiteLLM / a hosted gateway (OpenRouter alone) | LiteLLM is a large dependency and hides the routing logic this project is meant to teach; a hosted gateway is itself a single point of failure and its free tier is account-limited. |
| Retry the same provider with backoff | On free tiers the usual failure is a quota, not a blip; waiting burns user latency while another provider is idle. |
| Detect 429 by searching the error text | A number or the string "429" can appear in any error; the header is the contract. |
| Fixed per-minute counters reset every 60 s | Allows 2× the limit across a window boundary; the sliding window does not. |
| Count requests only on success | Providers count failed requests too, so the router would overshoot and trigger 429s. |
| Breaker that re-opens fully after a timer | A still-dead provider would receive full traffic again; a single half-open probe limits the damage. |
| Sleep-and-retry when all providers are busy | Blocks a user request for tens of seconds; failing fast with `Retry-After` lets the client decide. |

## Consequences

- Good: survives quota exhaustion and model deprecation; free by default; every behaviour is unit-tested with a fake clock.
- Good: `router.status()` gives an operator view (breaker state, budget use, last error per provider).
- Cost: answers can come from different models turn to turn, so quality/style varies. Mitigation: the per-task order keeps the strongest model first for tutoring, and each response records which provider served it.
- Limitation: state is in-process. Correct for one instance (the free-tier deployment). With several replicas each has its own windows and breakers, so the combined rate can exceed a provider's quota. The fix is to move `RateWindow` and breaker state into Redis (atomic `INCR` with expiry / a Lua script) — deferred until there is more than one instance (see ADR 0006).
- The rpm/tpm defaults are conservative guesses at free-tier limits; providers change them. They are config, and the `rate_limited` counters in `status()` show when they are too high.
