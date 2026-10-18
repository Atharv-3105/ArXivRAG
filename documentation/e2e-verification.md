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


# Data pipeline audit (2026-10-19)

The question this time was not "do the agents answer" but "is the data layer correct", checked against what is actually stored. Method: query Postgres and Qdrant, compare to the source HTML, and measure retrieval.

## What was correct

| Check | Result |
|---|---|
| Section structure of HTML-parsed papers | Real section titles, 605-716 distinct sections over 21 papers, no empty or generic titles. |
| Reference extraction recall | 100%: every bibliography entry in the source HTML is stored (16 of 16 papers). |
| Equation extraction | Matches the source's equation count (a consistent ratio against a raw class count that double-counts nested wrappers). |
| Formulas split across chunks | 1 of 1,749 chunks (0.1%) before, 0 after. |
| Postgres vs Qdrant | Chunk count equals point count, before and after re-ingest. |
| Dead-letter queue | The one failed paper is a legitimate 404 for a PDF, parked with a retry count. |
| Ingest idempotency and re-ingest | Re-ingest replaced papers in place, only after the new parse succeeded. |
| Hybrid retrieval | Fusion beats either branch alone from rank 3 (table in ADR 0015). |
| `/papers/missing` | After the graph rebuild it correctly lists the most-cited absent papers (Llama 3 x4, Qwen3 x3, LoRA x2), which guides corpus growth. |

## Defects found and fixed

| # | Defect | Fix |
|---|---|---|
| 1 | **32% of stored chunks overflowed MiniLM's 256-token window** and were silently truncated (about half of each lost). The earlier 800-character fix had not been measured. | Token-aware chunker with the model's own tokenizer, reserving the section path (ADR 0015). 0.1% overflow now. |
| 2 | **The citation graph was empty (0 edges) although 300 references carried arXiv ids.** Edges and contexts were only written at ingest, papers stored before Phase 4 never got them, and `reindex` only re-embeds, despite the architecture doc saying "rebuildable". | `python -m src.pipeline.rebuild_graph`: edges from stored references, contexts by re-parsing cached HTML, then influence. 365 edges and 526 contexts. |
| 3 | **A revised arXiv version was stored next to the old one** (3 of 21 papers duplicated: 24 rows, 215 duplicate chunks), so the same passage would be cited as two sources. | A new version supersedes older ones after the new parse succeeds; tested, including that a failed parse never deletes the old version. |
| 4 | **5 papers had no parse provenance** (`parse_method` NULL, no equations, no references): PDF-era rows the app could not flag as degraded. | Re-ingest: arXiv now serves HTML for them, so all 21 are HTML with LaTeX math. |
| 5 | **Every new database connection took 21 s** and `/health/ready` reported `database: timeout`, while Postgres was healthy. `localhost` resolved to IPv6 first and Docker dropped it. | `127.0.0.1` in the config, and a 5 s connect timeout in the engine. Runbook entry added. |
| 6 | **Citation-function labels were wrong about 3 times in 4** on real sentences (a hand-judged sample of 24). The offline eval had hidden this, because it uses hand-made sentences. | A sentence that cites a group is labelled neutral. Precision on the same kind of sample rose to about 54% with 5% of contexts labelled. |

## Still weak (recorded, not fixed)

1. **`critical` is unreliable** (about 2 of 9 correct in the second sample). It fires on sentences describing a problem in the field that cite a paper as support, not on criticism of that paper. `builds_on` is mostly right (about 11 of 16). Options: an LLM classification pass at ingest, or not showing `critical` until it is validated. Features that say "who criticises X" inherit this.
2. **Only 34% of references carry an arXiv id** (365 edges from 1,074 references). The rest cite venue papers; resolving them needs title or DOI matching (OpenAlex or Semantic Scholar are free), otherwise the graph stays sparse even with a large corpus.
3. **41 chunks are under 150 characters**: the remainder at the end of a section. Harmless but low value; folding a short remainder into the previous chunk would remove them.
4. **No retrieval regression test over the real corpus**: the question sets used here live in `.tmp`. They should become a checked-in eval (`src/eval/data`) that runs in the live suite.
5. **The graph is still mostly outside the corpus**: all 365 edges point to papers not ingested, so PageRank and prerequisites are not meaningful until the foundation list is ingested.
