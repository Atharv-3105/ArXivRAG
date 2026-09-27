# API

The full spec is [openapi.json](openapi.json), generated from the code (`python -m src.api.openapi_export`), so it can't drift. Interactive docs are served at `/docs` when `DEBUG=true`.

| Area | Endpoints | Auth |
|---|---|---|
| Auth | `POST /api/v1/auth/register`, `/login` | none (returns a bearer token) |
| Chat | `POST /api/v1/chat`, `/chat/stream` (SSE), `POST /api/v1/feedback` | bearer |
| Me | `GET /api/v1/me`, `PATCH /me/profile`, `GET /me/mastery`, `PUT /me/mastery/{concept}`, `GET /me/reviews`, `GET /me/audit`, `GET /me/export`, `DELETE /me` | bearer |
| History | `GET /api/v1/conversations`, `/conversations/{id}` | bearer |
| Papers | `GET /api/v1/papers/top`, `/missing`, `/{id}`, `/{id}/graph`, `/{id}/prerequisites`, `/{id}/citations` | public, per-IP limited |
| System | `GET /health`, `/health/live`, `/health/ready`, `/metrics` | public (`/metrics`: optional token) |
| UI | `GET /ui/graph?paper=<id>` | public |

**Errors:** always `{"error": {"code", "message", "trace_id"}}`. Validation errors are 400 with `details`, limits are 429 with `Retry-After`, and no LLM available is 503 with `Retry-After`.

**SSE events (`/chat/stream`):** `meta`, `intent`, `sources`, `provider`, `token`…, then `done` or `error` (see `src/api/routes/chat.py`).

**Chat request:** `{"message", "conversation_id"?, "action"?, "paper_id"?, "equation"?}`. `action` is one of `answer | hint | dont_understand | skip | review | read | next | equation`.
