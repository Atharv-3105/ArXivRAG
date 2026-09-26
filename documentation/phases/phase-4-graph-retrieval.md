# Phase 4: Graph and advanced retrieval

> Goal (plan §6): citation graph integration, graph-enhanced context, influence scoring, mastery-weighted retrieval, and a graph endpoint for the UI.
> Decisions: [ADR 0009](../adr/0009-graph-enhanced-retrieval.md) (this phase), [ADR 0003](../adr/0003-citation-graph-in-postgres.md) (graph in Postgres).

## 1. Data added at ingestion

```
arXiv HTML
  <cite class="ltx_cite"><a href="#bib.bib12">[12]</a></cite>   in-text citation
  <li id="bib.bib12" class="ltx_bibitem">… arXiv:2106.09685 …</li>    bibliography entry
        │
        ├── paper_citations   (citing → cited base id)                        Phase 1
        └── citation_contexts (citing, cited base id, sentence, function)     Phase 4
                                   function = critical | comparative | builds_on | neutral
after each run / reindex:
        paper_influence (PageRank over the in-corpus graph, in-degree)
```

Existing papers gain citation contexts with `python -m src.pipeline.runner --reingest …`. The HTML comes from the disk cache, so arXiv isn't hit again.

## 2. Retrieval with the graph

```
hybrid search (ADR 0005) ──► vector-only sources  (baseline, always kept)
        │
        └─► within GRAPH_TIMEOUT_SECONDS (1.5 s):
              influence(hit papers)
              re-rank by reader level      weak: background + influential · strong: methods + recent
              expand by intent             learn+weak → prerequisites
                                           critique   → what other papers say (critical/comparative)
                                           synthesize → citation-connected papers
            any error or timeout → baseline, graph_status = "failed"
```

`graph_status` (`ok` / `failed` / `off`) is in the `/chat` response and the SSE `done` event. The prompt (`chat-v3`) labels graph sources and asks for attribution.

## 3. Endpoints

| Path | Purpose |
|---|---|
| `GET /papers/top` | most influential papers (PageRank, 1.0 = average) |
| `GET /papers/missing` | most-cited arXiv papers that are not in the corpus: an ingestion to-do list |
| `GET /papers/{id}` | metadata, influence, reference and citation counts (also accepts a base id) |
| `GET /papers/{id}/graph?depth=1..2` | nodes and edges with each edge's strongest citation function |
| `GET /papers/{id}/prerequisites` | papers it builds on, nearest first, then most influential |
| `GET /papers/{id}/citations?function=critical` | the citing sentences about it |
| `GET /ui/graph?paper={id}` | force-directed viewer (D3); click an in-corpus node to recentre |

These endpoints are public and rate-limited per IP (`RATE_LIMIT_PUBLIC_PER_MINUTE`). Old-style ids (`hep-th/9802150v1`) work through `:path` routes.

## 3a. Visual check (pending)

The API behind the viewer is covered by tests. The page itself has only been checked to load: it's a static file and hasn't been visually verified against live data. That needs the live corpus once Docker is back.

## 4. Failure modes

| Failure | Behaviour |
|---|---|
| Graph query slow | cancelled by Postgres (`statement_timeout`) and by the per-turn budget → vector-only |
| Postgres down | graph → `failed`, the answer is still produced from vector results |
| Citation cycles | the path-array guard in the recursive CTE, and `depth ≤ 4` |
| Huge neighbourhood | node cap (`GRAPH_MAX_NODES`), with `truncated: true` in the response |
| PDF-parsed paper | edges only if references were parsed, and no contexts. Documented limitation. |

## 5. Acceptance criteria (plan §6, Phase 4)

| Criterion | Status |
|---|---|
| Graph queries return valid paths | Done: prerequisite paths start at the root, respect depth, and survive cycles (tests). |
| Hybrid retrieval improves relevance | **Hypothesis, not yet measured.** The mechanism is in place and the weights are config. Measuring it is the Phase 6 eval (the same questions with graph off and on). |
| Citation grounding increases | Graph sources are numbered like any other and resolved by the same `[n]` grounding. The rate is measured in Phase 6. |
| UI renders force-directed graph | The `/ui/graph` viewer is done. Its API is tested, but it has **not been visually verified with live data**. |
| Query timeouts, fallback to vector-only, payload limits | Done and tested (slow and broken store, caps). |
| Live SQL run | **Pending** (Docker). The recursive CTE is asserted structurally: cycle guard, depth bound, limit. |

## 6. Interview questions (with the answer you should be able to give)

1. **"How do you combine a knowledge graph with vector search?"** Keep the vector result as a baseline. Use the graph to re-rank (influence, reader level) and to *add* context the embedding can't see (prerequisites, what others say, related work). It's additive and time-boxed, so a graph failure can only remove extras.
2. **"Why PageRank and not citation count?"** Rank flows through citations, so the paper everything builds on beats a merely popular one. It's precomputed after ingestion because it only changes when the corpus does.
3. **"How do you write a safe recursive query over a cyclic graph?"** A path-array cycle guard, a depth bound, `DISTINCT ON` to keep each node's shallowest path, a `LIMIT`, and a statement timeout.
4. **"How would you detect that one paper criticises another?"** Recover the citing sentence (a LaTeXML cite links to a bibliography entry, which links to an arXiv id), then classify its function. Rules for precision now, a trained classifier once there's a labelled sample.
5. **"Why did your re-ranker initially misbehave?"** 1/(60+rank) is almost flat, so a 1.25× boost moved hits about 15 places. Use a steeper curve (k=10) and cap the boost. A property test ("the clear winner stays first") caught it.
6. **"How does personalisation affect retrieval?"** Mean mastery of the query's concepts gives weak, strong or unknown. Weak readers get background sections, influential papers and prerequisite expansion. Strong readers get methods and recent work.
7. **"How do you keep graph features from hurting latency?"** Precompute influence, time-box the per-turn graph work (1.5 s), set a statement timeout per query, and cap sizes. Measure p95 with and without the graph (Phase 6).
8. **"Why is the papers API public?"** Public data, no LLM cost. It still gets a per-IP limit and payload caps to protect a free-tier database. Mention `--proxy-headers` so the IP is the real client's.
9. **"How would you evaluate the graph's contribution?"** A/B on a fixed question set: vector-only versus graph, measuring nDCG or recall@k against labelled relevant chunks, citation grounding rate, and answer-judge scores. Only keep boosts that win.
