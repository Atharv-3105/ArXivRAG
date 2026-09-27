# ADR 0013: Free-tier deployment, CI/CD, rollback and secrets

- **Status:** Accepted (2026-09-27). Plan Phase 6 (CI/CD, blue-green, rollback, secret rotation, SLOs).
- **Files:** `.github/workflows/ci.yml`, `.github/workflows/ingest.yml`, `render.yaml`, `docker-compose.yml` (profiles `api`, `observability`), `Dockerfile`, `documentation/deployment.md`, `documentation/runbooks/`

## Context

The target is a shareable SaaS on free tiers: one small web container, a managed Postgres, a managed Qdrant and free LLM quotas. The plan asks for lint → type-check → test → build → push → deploy, manual approval for production, blue-green deploys, rollback and secret rotation.

## Decision

| Concern | Choice |
|---|---|
| **Build** | CI: lint, format, mypy, tests, **offline eval gate**, **gitleaks** secret scan over full history, then Docker build, image size reported, smoke test (imports, templates, *no torch in the image*) |
| **Artifact** | one image, pushed to GHCR on `main` with an **immutable `:<git-sha>` tag** plus `:latest` |
| **API host** | a container platform (Render blueprint provided); `/health/ready` as the health check |
| **Blue-green** | the platform's health-checked deploy: the new instance only takes traffic once `/health/ready` is 200, and the old one serves until then |
| **Manual approval** | a GitHub *environment* `production` with required reviewers gates the deploy step |
| **Rollback** | redeploy the previous `:<git-sha>` tag. Schema changes are additive only (`ADD COLUMN IF NOT EXISTS`), so an older image runs against a newer schema. |
| **Ingestion** | a scheduled GitHub Actions workflow (daily) instead of an always-on worker. Idempotent (ADR 0002), `concurrency` prevents overlapping runs, and a `reindex` step repairs anything left pending. |
| **Secrets** | platform secret stores and GitHub Actions secrets only. `.env` is git-ignored, and `render.yaml` marks every secret `sync: false`. gitleaks fails the build on a committed secret. |
| **JWT key rotation** | `SECRET_KEY_PREVIOUS`: new tokens use the new key, old tokens verify until they expire, then the previous key is removed |
| **Metrics exposure** | `/metrics` is protected by `METRICS_TOKEN` when public |

## Alternatives considered

| Option | Why not |
|---|---|
| Kubernetes + Argo Rollouts for blue-green | Far beyond one free container. The platform's health-gated deploy gives the same guarantee for one instance. |
| Always-on ingestion worker | A second paid process. Ingestion is a daily batch, and a scheduled workflow is free and leaves logs per run. |
| Alembic migrations now | Every schema change so far is additive and idempotent at startup. Alembic arrives with the first destructive change (rename or drop), with an expand/contract plan. |
| Mutable `:latest` only | You can't roll back to "the previous latest". Sha tags make rollback a one-field change. |
| Secrets in a committed, encrypted file (sops) | Key management overhead for a single-maintainer project. Platform secret stores are free and audited. |

## Consequences

- Good: a merge to `main` produces a tested, scanned, smoke-tested, immutable image. Rollback is choosing an older tag, and ingestion costs nothing when idle.
- Limitation: free web hosts sleep when idle. The first request after a sleep is slow (cold start plus model load) and would breach the latency SLO. Mitigations are in `documentation/deployment.md`: a keep-warm ping within the host's terms, or accept it and exclude cold starts from the SLO window.
- Not done here: no remote repository is configured and no accounts exist, so **nothing has been deployed**. The workflows and blueprint are ready and validated only as far as they can be without running them.
