# Qdrant down, empty, or with the wrong layout

**Symptoms:**
- `DegradedAnswers` fires, and users see "Paper search is temporarily unavailable" or "I couldn't find anything…";
- `/health/ready` shows `qdrant: error` (not required, so traffic continues);
- or startup fails with `VECTOR_SCHEMA_MISMATCH`.

**Diagnose**
1. `/health/ready`: is `qdrant` `ok`?
2. Logs: `Qdrant search failed` (outage) vs. `No sources` (empty or wrong collection).
3. Count points versus chunks. Points in the collection (Qdrant console) should equal `SELECT count(*) FROM paper_chunks` for indexed papers.

**Fix**

| Cause | Action |
|---|---|
| Outage / suspended free cluster | resume it in the console. Answers recover without a restart |
| Collection missing or wiped | `python -m src.pipeline.reindex --all`: rebuilds every vector from Postgres, with no re-download (ADR 0002) |
| `VECTOR_SCHEMA_MISMATCH` (e.g. embedding model changed) | set `QDRANT_COLLECTION` to a new name, run `reindex --all`, deploy, then delete the old collection |
| Papers stuck `pending` | `python -m src.pipeline.reindex` repairs only those |

**Verify:** the degraded share drops, and a chat turn returns sources with `graph_status: ok`.
