# Web app

Design and architecture: [ADR 0014](adr/0014-web-frontend.md).

## Run it

```bash
# Everything offline (in-memory stores, built-in papers, mock LLM unless keys are in .env):
cd frontend && npm install && npm run build && cd ..
python -m src.api.demo                 # http://localhost:8000

# Frontend development with hot reload (proxies /api to :8000):
python -m src.api.demo                 # terminal 1
cd frontend && npm run dev             # terminal 2 -> http://localhost:5173
```

In production the Docker image builds `frontend/` and FastAPI serves it at `/` (`FRONTEND_DIST`).

## Layout

| Path | What |
|---|---|
| `src/api/` | `client.ts` (fetch, token, errors), `stream.ts` (SSE over fetch), `types.ts` (API shapes) |
| `src/lib/render.ts` | answer text → blocks: markdown, KaTeX, `[n]` → sidenote links |
| `src/lib/terrain.ts` | the paper map: force layout by field, density contours, label placement |
| `src/components/` | `AnswerPage` (page + sidenotes), `Exercise` (the tutor's question), `Composer`, `Rail` |
| `src/pages/` | `Chat`, `Reviews`, `Reading`, `MapPage` (lazy), `Me`, `Login` |
| `src/styles/` | `tokens.css` (the Margin palette and type), `app.css` |

## Checks

```bash
npx tsc --noEmit && npx vitest run && npx vite build
```
