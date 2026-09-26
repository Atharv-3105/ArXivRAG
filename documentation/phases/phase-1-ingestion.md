# Phase 1 — Ingestion pipeline

> Goal (plan §6): idempotent arXiv fetch → parse → chunk → embed → index.
> Decisions: [ADR 0001](../adr/0001-html-first-ingestion.md) (HTML first), [ADR 0002](../adr/0002-postgres-source-of-truth.md) (consistency), [ADR 0003](../adr/0003-citation-graph-in-postgres.md) (citation graph).

## 1. What runs, in order

```
ArxivFetcher.fetch_papers(query)          arXiv Atom API, 1 request / ARXIV_REQUEST_DELAY_SECONDS
  └─ for each paper (failures isolated per paper)
       get_index_status(arxiv_id)          indexed → skip | pending → repair | none → continue
       PaperExtractor.extract(paper)        HTML tier → PDF tier, disk-cached
       TextChunker.chunk(sections)          1000 chars, 200 overlap, formulas atomic
       PgRepo.upsert_paper_and_chunks       ONE transaction: paper(pending) + chunks + equations
                                            + references + citation edges
       VectorStore.upsert_vectors           bge-small (384-d, cosine), wait=True
       PgRepo.mark_indexed                  status → indexed, DLQ row cleared
  ingestion_runs row                        status + metrics (parse_success_rate, html/pdf split…)
```

Run it:

```bash
docker compose up -d db qdrant
python -m src.pipeline.runner --query "cat:cs.LG AND ti:attention" --max-papers 20
python -m src.pipeline.reindex            # repair 'pending' papers
python -m src.pipeline.reindex --all      # rebuild Qdrant from Postgres (model/collection change)
```

## 2. Components and the reasoning behind them

### Fetcher (`fetcher.py`)
- **Politeness is a correctness requirement.** arXiv asks for about 1 request every 3 seconds, and when you exceed it, it answers `200` with an HTML error page instead of a 429. So the fetcher waits between requests *and* checks that a "PDF" really starts with `%PDF`.
- **Retry only what is retryable.** Timeouts, connection errors, 429 and 5xx are flagged `retryable` and go through `tenacity` with exponential backoff plus jitter. A 404 or a malformed query fails at once, because retrying a permanent error only wastes the rate budget.
- **Jitter** keeps many clients from retrying in lock-step after a shared outage (the thundering herd).

### Extractor and parsers (`extractor.py`, `html_parser.py`, `parser.py`, `layout.py`)
See ADR 0001. The key idea: **record provenance** (`parse_method`, `math_fidelity`, `parse_stats`) so quality can be measured per paper and the answer layer can be honest about degraded math.

### Chunker (`chunker.py`)
- Sliding window by characters (1000 / 200). Characters rather than tokens so the chunker has no tokenizer dependency. bge-small's 512-token limit is about 2,000 characters of English, so 1000 characters stays well inside it, even for LaTeX-heavy text, which tokenizes poorly.
- **Formulas are never split.** Half a `$…$` span is invalid LaTeX, and it gets embedded as noise.
- **Section path is embedded with the text** (`"3 Method > 3.2 Loss\n<content>"`). This is a cheap form of contextual retrieval: a chunk that just says "we set λ = 0.1" becomes findable by "loss weighting in method".
- **Chunk ids are deterministic** (a UUID5 of paper, section and position), which makes re-indexing overwrite points instead of duplicating them.

### Embeddings (`embedder.py`, `vector_db.py`)
- **`BAAI/bge-small-en-v1.5`** (384-d) rather than the plan's bge-large (1024-d). It runs on CPU on a free host, its vectors are 2.7× smaller, and the quality gap on retrieval benchmarks is small. The model is config (`EMBEDDING_MODEL` and `EMBEDDING_DIM`), and a switch is a `reindex --all`.
- Embeddings are normalized and compared with **cosine** distance.
- Encoding is CPU-bound, so it runs in `asyncio.to_thread`. It would otherwise block the event loop, and with it every other coroutine, including the API when the pipeline runs in-process.
- The Qdrant payload carries `parse_method` and `math_fidelity`, so retrieval can filter or down-rank degraded chunks without a Postgres join.

### Storage (`schema.sql`, `repositories.py`)
- **One transaction per paper** for everything in Postgres, so a half-written paper is never visible.
- `ON CONFLICT DO NOTHING` on every insert means two concurrent runs racing on the same paper cannot create duplicates. The loser's insert is a no-op and it reports `skipped`.
- The schema is idempotent (`CREATE … IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`) and runs at startup. We skipped Alembic for now: every change so far is additive. Alembic comes in when the first destructive change arrives.
- **Dead-letter queue** (`failed_papers`): it has no foreign key on purpose, because a paper that failed before it was stored must still be recordable. `retry_count` is incremented on each failure, and the row is cleared on success.
- **`ingestion_runs`**: one row per run, used as an audit trail and as the source for the "parse success rate > 90%" metric.

### Consistency across two stores
See ADR 0002: a `pending → indexed` status flag in the source of truth, idempotent vector writes, and a reconciler.

### Citations
See ADR 0003: arXiv ids are pulled from references with context-anchored regexes into `paper_citations`, keyed by base id, and written in the paper's own transaction.

## 3. Failure modes and what happens

| Failure | Behaviour |
|---|---|
| arXiv API timeout / 5xx / 429 | Retried with backoff and jitter. After that, the whole run fails and is recorded as `failed` in `ingestion_runs`. |
| One paper's HTML and PDF both fail, or the parse is below the quality gate | That paper goes to the DLQ (`failed++`) and the run continues. |
| Qdrant down mid-run | The paper stays `pending` in Postgres and goes to the DLQ. The next run or `reindex` re-embeds it from Postgres. |
| Process killed between Postgres commit and Qdrant ack | Same as above: `pending` is the durable to-do marker. |
| Process killed mid-Postgres transaction | Rolled back. Nothing is visible and the next run starts fresh. |
| Two runs at once | `ON CONFLICT DO NOTHING` plus deterministic chunk ids mean no duplicates. |
| `--reingest` with a worse parser result | The old data is deleted only *after* the new parse succeeded. |
| DLQ write itself fails | Logged, never fatal: the run continues. |

## 4. Acceptance criteria (plan §6, Phase 1)

| Criterion | Status |
|---|---|
| Ingests 100 papers without duplicates | Idempotency is covered by tests (skip, race, reingest, repair). A live 100-paper run has **not been done yet**: it needs Docker up. |
| Parse success rate > 90% | Now computed per run (`ingestion_runs.metrics.parse_success_rate`). The live measurement is pending, as above. |
| Embeddings validated | A count mismatch raises `VECTOR_DIM_ERROR`, and the dimension comes from config. |
| Graph edges count matches reference metadata | The extractor is unit-tested against real reference formats. Live check: edges versus references containing an arXiv marker (SQL below). |
| Dead-letter queue, retries with jitter, idempotent writes | Done and tested. |

Verification queries for the live run:

```sql
SELECT status, metrics->>'parse_success_rate', metrics FROM ingestion_runs ORDER BY id DESC LIMIT 5;
SELECT index_status, count(*) FROM papers GROUP BY 1;                 -- all 'indexed' after a clean run
SELECT count(*) FROM paper_chunks;                                    -- must equal Qdrant points_count
SELECT count(*) FROM paper_references WHERE text ILIKE '%arxiv%';     -- upper bound for edges
SELECT count(*) FROM paper_citations;
SELECT c.cited_base_id, count(*) AS cited_by, bool_or(p.arxiv_id IS NOT NULL) AS in_corpus
FROM paper_citations c LEFT JOIN papers p ON p.base_id = c.cited_base_id
GROUP BY 1 ORDER BY 2 DESC LIMIT 10;                                  -- what to ingest next
```

## 5. Interview questions (with the answer you should be able to give)

1. **"You write to Postgres and a vector DB. How do you keep them consistent?"**
   You can't have a transaction across both, so pick a source of truth (Postgres) and treat the vector store as a derived index. Write Postgres with a `pending` marker, write the vectors idempotently (deterministic ids), then flip the marker. A crash leaves `pending`, and a reconciler replays it. Name the general pattern: a transactional outbox, and say why a column was enough here.
2. **"Why not write the vectors first?"** Orphaned vectors can be returned by search but point at nothing, and nothing in the source of truth records that they exist. Gaps are safe; orphans are not.
3. **"How is ingestion idempotent?"** At three levels. The status check skips papers that are already done. `ON CONFLICT DO NOTHING` guards against races. Deterministic chunk UUIDs make vector upserts replays rather than duplicates.
4. **"Which errors do you retry?"** Only transient ones: timeout, connection, 429, 5xx. The retries use exponential backoff and jitter. Explain why jitter matters (the thundering herd) and why retrying a 404 is harmful.
5. **"Why HTML instead of PDF?"** The HTML carries the author's LaTeX in `alttext`, so there is no math reconstruction to do. Quote the measurement (99.54% of 9,227 formulas verbatim) and the fallback for papers without HTML.
6. **"How did you choose chunk size?"** Start from the embedding model's context (512 tokens, about 2,000 characters). Stay well below it, because LaTeX tokenizes badly. Use 20% overlap so a sentence split at a boundary still appears whole in one chunk. Keep formulas atomic. Then say how you would tune it: build a retrieval eval set and sweep sizes. That is Phase 6.
7. **"What is contextual chunk embedding and why do it?"** Prepend the section path so an isolated chunk's vector reflects where it lives. It is the cheap version of Anthropic's "contextual retrieval".
8. **"Why bge-small and not a bigger model or an API?"** It runs on CPU on a free host, with no per-token cost and no data leaving the box. The gap to bge-large is small. The model is configurable, and a rebuild is `reindex --all` *because* Postgres keeps the text.
9. **"Why a dead-letter queue without a foreign key?"** Failures can happen before the paper row exists. The DLQ is a log of attempts, not a child of a paper.
10. **"Why is the citation graph in Postgres when the plan says Neo4j?"** The queries are bounded 1–3 hop traversals (recursive CTE). Postgres gives one store, one transaction and one backup, and Neo4j Aura Free pauses when idle. Give the revisit triggers and the migration path (the table is already an edge list).
11. **"How do you extract arXiv ids from references without false positives?"** Only ids anchored to an arXiv marker count (a prefix, a URL or the DOI), with lookarounds so page ranges don't match. Key edges on the version-less base id.
12. **"How would you scale this to 1M papers?"** Split fetch, parse and embed into queue-connected workers (the status column becomes a real outbox or queue). Batch embeddings on a GPU. Partition `paper_chunks`. Use Qdrant scalar quantization. Use arXiv's bulk S3 or OAI-PMH instead of the search API.
13. **"How do you know the parser is good?"** It is measured, not eyeballed: HTML ground truth versus PDF output on real papers, a formula-preservation rate, and `parse_success_rate` per run. The quality gate sends junk to the DLQ instead of the index.
