# ADR 0002 — Postgres is the source of truth; Qdrant is a derived, rebuildable index

- **Status:** Accepted (2026-09-26)
- **Code:** `src/pipeline/runner.py` (`_process_paper`, `index_from_store`), `src/pipeline/reindex.py`, `src/storage/repositories.py` (`get_index_status`, `mark_indexed`, `load_processed`), `schema.sql` (`papers.index_status`)
- **Tests:** `tests/test_runner.py` (retry-after-failure, ordering, skip), `tests/test_repositories.py`

## Context

Every paper is written to two stores: Postgres (metadata, chunks, equations, references) and Qdrant (vectors). They cannot share a transaction.

The old pipeline wrote Postgres, then Qdrant, and on the next run skipped any paper already in Postgres. If Qdrant failed in between (a timeout, or the process being killed), the paper stayed in Postgres forever but was never searchable. A strict `xfail` test pinned this bug until it was fixed.

## Decision

1. **One source of truth.** Postgres holds everything needed to rebuild the vectors (chunk text and section path). Qdrant is a *derived index*, like a database index: it can be dropped and rebuilt without calling arXiv.
2. **Two-phase status flag instead of a distributed transaction.** A paper is inserted with `index_status = 'pending'` (in the same Postgres transaction as its chunks). It is flipped to `'indexed'` only after Qdrant acknowledges the upsert (`wait=True`).
3. **The next run repairs.** For each fetched paper:

   | Postgres state | Action |
   |---|---|
   | not stored | download, parse, write Postgres as `pending`, write Qdrant, mark `indexed` |
   | `pending` | a previous run died between the writes: re-embed from Postgres (no download, no parse) |
   | `indexed` | nothing to do |

4. **Idempotent vector writes.** Chunk ids are deterministic UUIDs, so replaying the Qdrant upsert overwrites points instead of duplicating them. That is what makes "retry on `pending`" safe.
5. **A reconciler for papers the next query will never fetch again:** `python -m src.pipeline.reindex` repairs every `pending` paper, and `--all` rebuilds the whole collection (used for embedding-model or collection-schema migrations).
6. **Success clears the dead-letter queue.** `mark_indexed` deletes the paper's `failed_papers` row in the same transaction.
7. **Every run is recorded** in `ingestion_runs` with its status (`succeeded` / `partial` / `failed`) and metrics, including the parse success rate.

## Alternatives considered

| Option | Why not |
|---|---|
| Two-phase commit / XA across Postgres and Qdrant | Qdrant does not support it. |
| Write Qdrant first, then Postgres | A crash leaves vectors that point at nothing, and they are invisible to any Postgres-based check. Orphans in the derived store are worse than gaps. |
| Transactional outbox + worker | The correct pattern at scale. Here it would mean a queue and a worker process on a free tier that allows one process. The status column is the same idea (a durable to-do marker in the source of truth) with no extra moving parts. |
| Compare counts between stores on every run | O(corpus) per run, and it still does not say *which* chunks are missing. |
| Delete the Postgres row when Qdrant fails | It throws away a good parse, and a crash (as opposed to an exception) never runs the delete. |

## Consequences

- Good: a crash at any point converges on the next run or reindex, and the only redundant work is embeddings.
- Good: model and collection migrations are a `reindex --all`, not a re-crawl.
- Rule for readers: search results can be hydrated from Postgres, but a `pending` paper can be in Postgres without being searchable. Counts should be taken on `index_status = 'indexed'`.
- Existing rows default to `'indexed'`, because the old pipeline only left rows it had fully written.
