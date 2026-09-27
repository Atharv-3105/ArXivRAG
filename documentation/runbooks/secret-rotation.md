# Secret rotation

**When:** on schedule (e.g. every 90 days), when a maintainer leaves, or **immediately** on a suspected leak (a gitleaks alert, a key in logs, provider abuse notices).

## JWT signing key (`SECRET_KEY`), with no forced logout
1. Generate: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
2. Set `SECRET_KEY_PREVIOUS` to the **current** key and `SECRET_KEY` to the new one, then deploy. New tokens use the new key, and existing tokens keep verifying.
3. After `ACCESS_TOKEN_TTL_MINUTES` (default 24 h), every old token has expired. Remove `SECRET_KEY_PREVIOUS` and deploy.
4. **Leak instead of routine:** skip step 2's window (don't set `SECRET_KEY_PREVIOUS`). Everyone logs in again, which is the point.

## Provider and database credentials
| Secret | Rotate in | Then |
|---|---|---|
| LLM keys (`GROQ_API_KEY`, …) | provider console: create new, revoke old *after* deploy | update the host secret, deploy; `python -m src.llm.check` |
| `DATABASE_URL` password | database console | update the host **and** the Actions secret (ingestion) |
| `QDRANT_API_KEY` | Qdrant Cloud console | same two places |
| `METRICS_TOKEN` | generate new | update the host and the Prometheus/Grafana scrape config |

## After a leak
- Revoke first, investigate second.
- If the secret was committed: rotating is mandatory, because rewriting git history doesn't un-leak a pushed secret. Then clean history if needed.
- Check provider usage for abuse in the exposure window.
