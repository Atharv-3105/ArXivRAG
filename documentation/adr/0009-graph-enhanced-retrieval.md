# ADR 0009: Graph-enhanced retrieval, citation contexts and influence

- **Status:** Accepted (2026-09-26). Builds on ADR 0003 (graph in Postgres) and ADR 0005 (hybrid retrieval).
- **Code:** `src/pipeline/html_parser.py` (`_citation_contexts`), `src/pipeline/citations.py` (`citation_function`, `context_rows`), `src/graph/` (`store.py`, `influence.py`), `src/agents/graph_retrieval.py`, `src/api/routes/papers.py`, `src/api/static/graph.html`
- **Tests:** `tests/test_citation_contexts.py`, `tests/test_graph.py`

## Context

Plan Phase 4 wants citation traversal, prerequisite chains, influence scoring, critique detection, graph and vector fusion, mastery-weighted retrieval, and a graph endpoint for a force-directed view. It also wants query timeouts and a fallback to vector-only retrieval when the graph fails.

Two facts shaped the design:
1. An edge list (ADR 0003) says *that* B cites A, not *how*. "Who criticises A" needs the citing sentence.
2. LaTeXML links every in-text `<cite>` to its bibliography entry (`#bib.bib12`), and Phase 1 already resolves bibliography entries to arXiv ids. So the citing sentence *is* recoverable, for HTML-parsed papers.

## Decision

### Citation contexts
- The parser keeps the sentence around each in-text cite, with its bibliography keys and section anchor. Rows are written to `citation_contexts` in the paper's own transaction, only for keys that resolve to an arXiv id.
- **Citation function**, by rules (after Teufel et al.): `critical` > `comparative` > `builds_on` > `neutral`. For precision, "however" alone is not criticism, and ambiguous words ("ignore", "at the cost of") were removed from the lexicon.

### Influence
- **PageRank** over the in-corpus graph, scaled so 1.0 is average, plus in-degree. It is recomputed after every ingestion run and reindex (`python -m src.graph.influence`), stored in `paper_influence`, and never computed per request.
- Why PageRank rather than raw citation count: rank flows through citations, so the paper everyone *builds on* outranks the one merely cited often (the test corpus shows it: LoRA is cited three times, but the Transformer it builds on ranks first).

### Queries (`src/graph/store.py`)
| Need | Query | Bound |
|---|---|---|
| Prerequisite chain | recursive CTE over CITES into the corpus, path-array cycle guard, `DISTINCT ON` shallowest | depth ≤ 4, `LIMIT` |
| Who criticises / compares | `citation_contexts` by cited base id and function | `LIMIT` |
| Related papers | union of one-hop in and out neighbours | `LIMIT` |
| Neighbourhood for the viewer | BFS, one query pair per hop | depth ≤ 2 (API), nodes ≤ `GRAPH_MAX_NODES` |
| Ingest next | cited base ids with no corpus paper, by citing count | `LIMIT` |

Every query runs in a transaction with `SET LOCAL statement_timeout`.

### Retrieval fusion (`src/agents/graph_retrieval.py`)
1. **Baseline first:** the vector-only sources are computed and kept.
2. Within `GRAPH_TIMEOUT_SECONDS`: fetch influence, **re-rank by reader level**, then **expand by intent**:
   - learn + weak reader → the best chunks of the top result's prerequisites,
   - critique → critical and comparative sentences *about* the top results, from other papers,
   - synthesize → the best chunks of citation-connected papers.
3. On any exception or timeout the baseline stands, and `graph_status` says `failed`.

The prompt labels expanded sources (`prerequisite`, `what another paper says`, `related via citations`), and the model is told to attribute citation contexts rather than state them as fact.

### Re-rank maths
The score is `min(boost, 1.5) / (10 + rank)`. The first version used RRF's `1/(60 + rank)`, a curve so flat that a 1.25× section boost jumped about 15 positions and effectively reordered everything. A test ("boosts cannot leapfrog a much better hit") caught it. With k = 10 and the cap, a boost moves a hit past at most its few nearest neighbours.

| Reader | Boosts |
|---|---|
| weak (mean mastery < 0.35, or beginner) | background-type sections × 1.25; influence × (1 + 0.1·PageRank) |
| strong (> 0.7, or advanced) | method/experiment sections × 1.25; published in the last 2 years × 1.1 |
| unknown | influence × (1 + 0.05·PageRank) |

### API
Read-only and public (public data, no LLM cost), with a per-IP limit and size caps: `/papers/top`, `/papers/missing`, `/papers/{id}`, `/papers/{id}/graph`, `/papers/{id}/prerequisites`, `/papers/{id}/citations?function=`. There's also a static D3 force-directed viewer at `/ui/graph`.

## Alternatives considered

| Option | Why not |
|---|---|
| Neo4j + Cypher | ADR 0003. Nothing here needs more than bounded CTEs, and influence is precomputed. |
| An LLM or SciBERT classifier for citation function | Better recall, but it costs quota or a model per sentence at ingest. Rules are explainable and free. Revisit with a labelled sample in Phase 6. |
| Semantic Scholar "influential citations" API | External rate limits on the ingest path. Useful later as enrichment. |
| PageRank per request | O(E) per question. Scores only change at ingestion. |
| Fuse graph neighbours into the vector query as a filter | It would *replace* results instead of adding context, and a graph outage would break retrieval. Additive expansion with a baseline keeps failures harmless. |
| RRF-style k = 60 for the re-rank | Measured by a test to be too flat (above). |

## Consequences

- Good: "who criticises X", "what does X build on", "what should I ingest next" are all answerable from our own data, and the answer quotes the actual citing sentence.
- Good: graph trouble can only remove the extra context, never the answer.
- Limitation: citation contexts exist only for HTML-parsed papers, and only for references that carry an arXiv id. Existing papers need `--reingest` (served from the disk cache) to gain contexts.
- Limitation: "improves relevance" is a hypothesis until the Phase 6 eval set measures it (baseline versus graph on the same questions). The re-rank weights are config (`RERANK_*`, `MASTERY_*`) for exactly that experiment.
