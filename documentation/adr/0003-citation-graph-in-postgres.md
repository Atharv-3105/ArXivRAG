# ADR 0003 — Citation graph in Postgres instead of Neo4j (deviation from plan §3/§5)

- **Status:** Accepted (2026-09-26). This is a deviation from the plan, which lists Neo4j.
- **Code:** `src/pipeline/citations.py` (edge extraction), `schema.sql` (`paper_citations`, `papers.base_id`), `repositories.py`
- **Tests:** `tests/test_citations.py`, `tests/test_repositories.py`

## Context

The plan wants `(Paper)-[:CITES]->(Paper)` edges, prerequisite chains and influence scores, backed by Neo4j. The product must run on free tiers, and each extra managed service is another free quota to fit inside, another thing that can go to sleep, and another backup to take.

These are the queries we actually need, with depth capped by `GRAPH_MAX_DEPTH` (≤ 4):

- The papers this paper cites (1 hop), and the papers that cite this paper.
- Citation chains up to depth 2–3, for "read this first" prerequisites.
- Influence: in-degree inside our corpus, optionally weighted by recency.

These are shallow, bounded traversals over a graph of thousands to tens of thousands of edges. Postgres handles that with indexed joins and a `WITH RECURSIVE` query.

## Decision

1. **The edge table is `paper_citations(citing_id → papers, cited_base_id, ref_ordinal)`**, with primary key `(citing_id, cited_base_id)`. The cited paper **need not be in the corpus**: that lets us rank *un-ingested* papers by how often our corpus cites them, which is a ready-made "what to ingest next" signal.
2. **Base ids.** References almost never name a version, so edges point to the base id (`2401.12345`). `papers.base_id` is a generated column, `regexp_replace(arxiv_id, 'v[0-9]+$', '')`, and it is indexed, so the join is exact.
3. **Edges come from the parsed bibliography.** The arXiv API returns no references. Only ids next to an arXiv marker (`arXiv:`, `arxiv.org/abs|pdf|html/`, the `10.48550/arXiv.` DOI, or `abs/` in a reference that says arXiv) count. A bare `2401.12345` is as likely to be a page range. Self-citations to another version of the same paper are dropped.
4. **Edges are written in the same transaction as the paper**, so a paper is never stored without its edges, and `ON DELETE CASCADE` removes them with it.

## Alternatives considered

| Option | Why not (for now) |
|---|---|
| Neo4j Aura Free | An extra service with node and relationship caps, and it pauses when idle. It also needs a second consistency story (a third store to keep in sync, see ADR 0002). None of our queries need Cypher's strengths (deep or variable paths, graph algorithms). |
| Self-hosted Neo4j | A JVM service needs about 1 GB of RAM, which is more than the whole free app host. |
| Semantic Scholar / OpenAlex API for references | Better recall (they resolve non-arXiv references too), but it is a runtime dependency with its own rate limits. It is a good *enrichment* step later, not the ingest path. |
| Store edges in Qdrant payloads | Qdrant is not built for joins or traversal, and edges would then live in the derived store. |

## Revisit when

- Traversals need depth greater than 4, path ranking, or graph algorithms (PageRank, community detection) at interactive latency.
- The edge count reaches millions and recursive CTEs show up in p95 latency.

The migration path is cheap: `paper_citations` is already an edge list and can be bulk-loaded with `LOAD CSV`.

## Follow-up

ADR 0009 adds citation *contexts* (the citing sentence and its function) and precomputed PageRank on top of this edge table.

## Consequences

- Good: one database, one backup, one transaction. Graph writes and paper writes are atomic.
- Cost: recall is limited to references that carry an arXiv id. Journal-only references produce no edges. PDF-tier papers have no structured bibliography yet, so they produce no edges.
- The plan's Phase 1 criterion, "graph edges count matches reference metadata", is measured as edges versus references that contain an arXiv marker.
