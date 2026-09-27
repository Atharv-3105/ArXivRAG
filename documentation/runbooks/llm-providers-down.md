# LLM providers failing or slow

**Symptoms:** `LLMProvidersUnavailable`, `AllBreakersOpen` or `SlowFirstToken` fires. Users get 503 + `Retry-After`, "(Demo mode)" answers (every real provider down, so the mock answered), or long waits.

**Diagnose**
1. Dashboard: *LLM calls by provider and outcome*. The outcome says why:
   - `rate_limited`: quota;
   - `auth`: a bad or revoked key;
   - `transient`: provider outage or timeouts;
   - `bad_request`: a model was renamed or deprecated.
2. *Circuit breakers open*: which providers the router has taken out of rotation.
3. `python -m src.llm.check` (with the production env) smoke-tests each configured provider.

**Fix**

| Cause | Action |
|---|---|
| `auth` | rotate the key in the secret store and redeploy (breakers reopen after recovery) |
| `rate_limited` on one provider | nothing: the router fails over. If all are limited, add another free provider key, or lower `FREE_DAILY_MESSAGES` |
| `bad_request` after a model change | update the model name in config (`GROQ_MODEL`, `OPENROUTER_MODELS`, …) and redeploy |
| `transient` | wait. The breakers probe once per recovery window (half-open) and recover by themselves |
| Slow first token | move a faster provider earlier in `LLM_TASK_ORDERS` for the `tutor`/`default` tasks |

**Verify:** breakers return to closed, and the success share and first-token p95 recover on the dashboard.
