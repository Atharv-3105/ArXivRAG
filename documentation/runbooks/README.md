# Runbooks

Every alert in `ops/prometheus/alerts.yml` names one of these. Each has the same shape: **symptoms → diagnose → fix → verify**. Start from the alert's runbook, and use the trace id from a user report (it's in the error body, the `X-Trace-ID` header and on each stored message) to find the exact request in the logs.

| Runbook | Alerts |
|---|---|
| [deploy-and-rollback](deploy-and-rollback.md) | any regression right after a deploy |
| [llm-providers-down](llm-providers-down.md) | LLMProvidersUnavailable, AllBreakersOpen, SlowFirstToken |
| [database-down](database-down.md) | HighErrorRate, GraphRetrievalFailing, readiness 503 |
| [qdrant-down-and-reindex](qdrant-down-and-reindex.md) | DegradedAnswers, schema-mismatch startup failure |
| [quota-and-rate-limits](quota-and-rate-limits.md) | users hitting 429s, free-tier quota exhaustion |
| [memory-queue-backlog](memory-queue-backlog.md) | MemoryQueueBacklog |
| [ingestion-failures](ingestion-failures.md) | failed ingestion workflow, papers stuck `pending`, DLQ growth |
| [guardrail-alerts](guardrail-alerts.md) | GuardrailRejections |
| [secret-rotation](secret-rotation.md) | scheduled rotation, suspected leak |
