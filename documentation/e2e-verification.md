# End-to-end verification on the real stack (2026-10-18)

Everything before this was verified with the offline mock LLM, in-memory stores and fakes. This run used the real thing: Postgres 16 and Qdrant in Docker, Groq, Gemini and OpenRouter, a browser, and 21 ingested papers (the corpus from earlier ingest runs; the 236-paper foundation list in [ADR 0015](adr/0015-curated-corpus-and-minilm.md) was not ingested).

Repeatable with `python -m src.eval.e2e` (all scenarios) or `--only tutor,critic`. It registers throwaway users and erases them through `DELETE /me` at the end. `src/eval/e2e.py` is the harness and `src/eval/e2e_scenarios.py` holds the scenarios.

## Result

| Area | Verified | Notes |
|---|---|---|
| Schema migration | Yes | An old Phase 1 database (5 tables) was upgraded in place to 16 by the idempotent schema; the 21 papers kept their data. |
| Reindex from Postgres | Yes | `reindex --all` rebuilt 1,749 points into the MiniLM hybrid collection with no downloads, which is the "Postgres is the source of truth" design working. |
| Accounts, validation, ownership | 22/22 | Typed 400s, no user enumeration, other users' conversations are 404. |
| Rate limit, outage | 4/4 and 11/11 | A burst gets 429 with `Retry-After`. With every LLM key invalid: a typed `LLM_UNAVAILABLE` error, a 503 with `Retry-After`, no half-saved turn, and non-LLM paths keep working. |
| Guardrail | Yes | Off-topic refused with no LLM call; injection flagged and nothing leaked; smalltalk and acknowledgements answered. |
| Grounding | Yes | Every `[n]` pointed at a retrieved source. A famous method that is not in the library (LoRA) got "the provided sources do not explain how LoRA works", not an invention. |
| Synthesizer, critic, advisor, reading, equations | Yes | Comparison tables, weaknesses and strengths, deterministic ranking for identical state, section navigation. |
| Tutor | Mostly | Explain, one question, state persisted, hints counted and capped, "I don't understand" lowers the level, graded answers, skip. Graded verdicts were not re-confirmed after the quota ran out (see below). |
| Persistence, privacy | 14/14 | Export, profile, mastery, deletion (the old token and password stop working). |
| Browser | Yes | Sign-up, streaming, typeset maths, sidenotes, reload of stored conversations. |

## Defects found and fixed

Each one was invisible to the offline suite.

| # | Defect | Cause | Fix |
|---|---|---|---|
| 1 | Assistant messages lost their citations, agent, provider, prompt version and checks when saved | A multi-row `INSERT ... VALUES` takes its columns from the **first** row (the 3-key user message), so the 8-key assistant row's extra columns were dropped. The in-memory store cannot show this. | One insert per message; a regression test on the compiled statements. |
| 2 | An on-topic question was refused ("RBS-Attention ... prefill") | The word tokenizer kept hyphens inside words, so `rbs-attention` was one token and never matched `attention`. | Split compounds into parts; added missing terms; tested on 10 on-topic and 5 off-topic phrasings. |
| 3 | No citations or sidenotes, and a false "carry no citation" notice | gpt-oss on Groq cites as `【2】`, attached to the word. The citation pattern needs `[2]` after a non-word character. | `OutputGuard` rewrites markers and inserts the space, across stream chunk boundaries. Frontend also repairs stored messages. |
| 4 | An answer was 16,103 spaces and 29 other characters | Gemini padded a Markdown table until it hit the token limit. | The guard collapses padding and, because it holds back the first 120 characters, detects a degenerate start **before** anything is shown, so the router can still fail over. Prompt rules ask for compact tables. |
| 5 | Equations printed as raw TeX | Models write `\(..\)` and `\[..\]`; markdown eats the backslash. | The renderer rewrites them to `$` form before markdown. |
| 6 | The OpenRouter fallback was dead | Both default free models had been retired ("unavailable for free"). | Replaced with working chat models. `openrouter/free` was rejected: its auto-router answered "User Safety: safe" from a content-safety model. |
| 7 | A 3rd hint could be requested (3 of max 2) | The cap was enforced only on the wrong-answer path; the explicit hint action had none. | Over budget, the tutor explains more simply instead; regression test. |
| 8 | Whitespace-only messages were accepted and spent quota | `min_length=1` before stripping. | Strip first, then validate. |
| 9 | `OPEN_ROUTER_API_KEY` in `.env` was silently ignored | The setting is `OPENROUTER_API_KEY`. | Both spellings are accepted. |
| 10 | The provider smoke test reported a healthy Groq as failing | A 20-token budget is consumed by a reasoning model's hidden thinking. | Budget raised, with a comment. |

A failed trailer check now logs the provider and the raw tail of the output, so the next failure leaves evidence.

## Known limitations (not fixed, recorded)

1. **Free-tier capacity is the dominant risk.** Testing exhausted OpenRouter's daily free-model cap, Gemini's quota and Groq's per-minute limit. When all are down the app degrades correctly (typed error, nothing stored), but a learner sees no answer. The `.env` has a Cerebras line with no key: adding one (free, fast) is the cheapest improvement, and OpenRouter credits lift its daily cap.
2. **Fallback models comply less.** `nemotron` took up to 33 s and sometimes missed the one-question rule or returned no citations. The product measures this (`checks`) but by design never retries.
3. **Concepts outside the vocabulary leave no learning trace.** "RBS-Attention", "CoLearn" and "score centering" map to no concept, so no mastery or review is recorded for them ([ADR 0008](adr/0008-learner-model-and-memory-pipeline.md) accepted this). A fallback of "the top cited paper as the concept" would fix it.
4. **The domain gate is English-only**: a Hindi question about transformers is refused.
5. **`paper_id` is for reading and equations only**; it does not restrict retrieval for questions.
6. **No graph results to verify**: this corpus has 0 citation edges, so PageRank, prerequisites and the map's trails were exercised only on empty data.
7. **Polish**: stored conversations and the recent-conversations list show a blank state for a moment before loading.

## Not yet verified (blocked by quota, rerun when it resets)

Tutor grading quality with a strong model after the fixes, completing all 4 tutor steps, review scheduling from graded answers, and the critic/synthesizer output-contract rates over many turns. `python -m src.eval.e2e --only tutor,critic,synthesizer` covers them.
