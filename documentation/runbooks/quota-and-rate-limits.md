# Rate limits and quotas

**Symptoms:** users report 429s. Two codes:
- `RATE_LIMITED`: more than `RATE_LIMIT_PER_MINUTE` requests in the last minute;
- `DAILY_QUOTA_EXCEEDED`: `FREE_DAILY_MESSAGES` reached, resetting at 00:00 UTC.

Also `arxivmind_rate_limited_total` rising.

**Diagnose:**
- the `kind` label: `per_minute`, `daily_quota` or `public_ip` (the papers API);
- is it one user or many? Grep logs by trace id or user id. A single heavy user is the limiter working as intended.

**Fix**

| Situation | Action |
|---|---|
| Legitimate heavy user (you, a demo) | add their email to `ADMIN_EMAILS` (exempt) |
| Limits too tight for normal use | raise `FREE_DAILY_MESSAGES` only if the LLM quotas allow it: per-user limits exist to protect the shared free LLM budget |
| Abuse of the public papers API | lower `RATE_LIMIT_PUBLIC_PER_MINUTE`. Make sure the host passes the real client IP (uvicorn `--proxy-headers`) |
| Several instances | the per-minute window is per process (ADR 0006): move it to Redis before scaling out |

**Verify:** 429s return to the expected background level.
