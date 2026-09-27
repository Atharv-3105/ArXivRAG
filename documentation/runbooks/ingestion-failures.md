# Ingestion failures

**Symptoms:**
- the scheduled `ingest` workflow is red;
- no new papers for more than 26 h (the freshness SLO);
- papers stuck `pending`;
- `failed_papers` (the DLQ) growing.

**Diagnose** (SQL):
```sql
SELECT status, metrics FROM ingestion_runs ORDER BY id DESC LIMIT 5;        -- run outcome, parse rate
SELECT index_status, count(*) FROM papers GROUP BY 1;                        -- anything pending?
SELECT failure_reason, count(*) FROM failed_papers GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
```

**Fix**

| Cause | Action |
|---|---|
| arXiv API errors / 403 / HTML-instead-of-PDF | rate limit (`ARXIV_REQUEST_DELAY_SECONDS` ≥ 3), retries are automatic; re-run the workflow later |
| TLS handshake errors in the container | the Debian 12 pin in the Dockerfile exists for this (arXiv rejects a newer OpenSSL handshake); don't unpin |
| Papers `pending` | `python -m src.pipeline.reindex` (Qdrant was unreachable mid-run) |
| DLQ: parse quality (`PARSE_LENGTH_ERROR`, …) | expected for some papers (scans, very short notes); inspect a few; don't lower `PARSE_MIN_WORDS` blindly |
| Workflow secrets missing | check the Actions secrets `DATABASE_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, `SECRET_KEY` |

**Verify:** the next run is `succeeded`, `parse_success_rate` > 0.9, and nothing is left `pending`.
