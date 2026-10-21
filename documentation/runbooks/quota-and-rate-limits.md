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


## Sign-in lockouts and one reply at a time

- **`429` with "Too many failed sign-in attempts"**: failed sign-ins are counted per (email, IP) (`LOGIN_MAX_FAILURES`, default 8), per IP (`LOGIN_IP_MAX_FAILURES`, 40) and per email from anywhere (`LOGIN_EMAIL_MAX_FAILURES`, 100) over `LOGIN_LOCKOUT_SECONDS` (900). A success clears the pair. State is in process memory, so a restart clears it. **Behind a proxy run uvicorn with `--proxy-headers`**, otherwise every client shares the proxy's address and the per-IP limit locks everyone out together.
- **Too many sign-ups**: `REGISTER_MAX_PER_IP_PER_HOUR` (20).
- **`409 TURN_IN_PROGRESS`**: a second message arrived while a reply was still being written in the same conversation (two tabs, a double submit). The claim is released when the turn ends, fails or the client disconnects, and expires after 180 s.
