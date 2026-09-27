# Deployment (free tier)

> Decisions: [ADR 0013](adr/0013-deployment-and-operations.md). Nothing is deployed yet: there's no git remote and no accounts. This is the procedure.
> Free-tier limits change often. Check each provider's current limits before relying on the numbers here.

## Pieces

| Piece | Free option (example) | Notes |
|---|---|---|
| Code + CI + image registry | GitHub + Actions + GHCR | CI pushes `ghcr.io/<owner>/<repo>:<sha>` on `main` |
| Web API | a container host with a free tier (e.g. Render; `render.yaml` provided) | health check `/health/ready`, command in the blueprint |
| Postgres | a managed free Postgres (e.g. Neon or Supabase) | set `DATABASE_URL`; the schema is created at startup |
| Vector DB | Qdrant Cloud free cluster | set `QDRANT_URL`, `QDRANT_API_KEY` |
| LLMs | Groq / Gemini / Cerebras / OpenRouter free keys | any subset; the router fails over (ADR 0004) |
| Ingestion | `.github/workflows/ingest.yml` (daily) | needs the same DB/Qdrant secrets |
| Metrics | Grafana Cloud free tier scraping `/metrics`, or the compose profile locally | protect with `METRICS_TOKEN` |

## First deploy

1. Push the repo to GitHub. CI builds and pushes the image on `main`.
2. Create Postgres and Qdrant free instances and note their URLs and keys.
3. Generate a secret: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
4. Put `SECRET_KEY`, `DATABASE_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, at least one LLM key and `METRICS_TOKEN` in the host's secret store and in the GitHub Actions secrets (the ingestion workflow needs the DB/Qdrant ones).
5. Set `render.yaml`'s image to your `ghcr.io/<owner>/<repo>:<sha>` and create the service from the blueprint.
6. Seed the corpus: run the ingestion workflow manually, or `python -m src.eval.seed` locally against the cloud URLs.
7. Check it: `curl https://<host>/health/ready`, then a chat through the API.
   Owner account: set `SEED_ADMIN_EMAIL` and `SEED_ADMIN_PASSWORD` as secrets. Every start creates the account as an admin, or promotes an existing account to admin. It never resets an existing password, so rotate the password in the app. Admins are exempt from the rate limit and the daily quota. There is deliberately no passwordless or "debug login" path: that would be a backdoor in every deployed copy of the image.
8. Protect `main` and add a GitHub *environment* `production` with required reviewers, for manual approval of deploys.

## Deploy / rollback

- **Deploy:** point the service at the new `:<sha>` tag. The host starts it, waits for `/health/ready`, then switches traffic (blue-green for one instance).
- **Rollback:** point it back at the previous `:<sha>`. This is safe because schema changes are additive only.
- Full steps and checks: [runbooks/deploy-and-rollback.md](runbooks/deploy-and-rollback.md).

## Cold starts

Free web hosts sleep idle services. The first request afterwards pays the container start plus loading the ONNX models (bge-small and BM25, cached in `MODEL_CACHE_DIR`), which breaches the latency SLO for that request. The options:
- accept it and exclude cold starts from the SLO (documented in `slo.md`);
- an external keep-warm ping, only if the host's terms allow it;
- a paid instance, when there are real users.

## Local, everything

```bash
docker compose up -d db qdrant
docker compose --profile api up -d            # API on localhost:8000
docker compose --profile observability up -d  # + Prometheus :9090, Grafana :3001
docker compose --profile pipeline run --rm pipeline   # ingest
```
