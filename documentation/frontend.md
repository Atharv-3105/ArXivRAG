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
| `src/lib/render.ts` | answer text → blocks: markdown, KaTeX, `[n]` → sidenote links; repairs `\(..\)` maths and `【n】` citations |
| `src/lib/trace.ts` | the agent pipeline behind one answer, as display data (route, retrieval, writer, checks, timings) |
| `src/lib/useResource.ts` | fetch with loading / failed / loaded and race safety; `arxiv.ts`, `theme.ts`, `useTitle.ts`, `pointer.ts` |
| `src/lib/terrain.ts` | the paper map: force layout by field, density contours, label placement |
| `src/components/` | `AnswerPage` (page + sidenotes), `TracePanel` ("How this was answered"), `AnswerActions` (copy, rate), `Exercise` (the tutor's question), `Composer`, `Status` (loading / error), `Rail` |
| `src/pages/` | `Chat`, `Reviews`, `Reading`, `MapPage` (lazy), `Me`, `Login` |
| `src/styles/` | `tokens.css` (the Margin palette and type), `app.css` |

## Checks

```bash
npx tsc --noEmit && npx vitest run && npx vite build
```


## Behaviour worth knowing

Found by using the app on the real stack, then fixed (see [e2e-verification.md](e2e-verification.md)):

- **Three different states, never one message.** Every fetched screen shows *loading* (a skeleton), *failed* (the reason and Try again) or *empty* (a real empty message). It used to show "The library is empty" when the request had failed.
- **A turn can be stopped and retried.** Stop aborts the stream; the server does not save a cancelled turn, and the page says so. A failed or stopped answer offers Try again, which re-sends the same request without duplicating the question.
- **A cut connection is not an answer.** A stream that ends without its `done` event is reported as interrupted; it used to look like a complete (short) answer.
- **Errors are written for people.** `friendlyError` maps backend codes (`LLM_UNAVAILABLE`, `RATE_LIMITED`, `DAILY_QUOTA_EXCEEDED`, ...) to plain wording; the server's own text is for logs.
- **Following a streaming answer** is automatic until the reader scrolls up; a Latest button returns. The landing screen never auto-scrolls (it hid the heading on phones).
- **Sessions that end say why.** A rejected token returns to sign-in with a notice and, after signing in, back to the screen you were on (only ever a path inside this app).
- **How this was answered** shows what the agents did for each answer: how the question was read, what was retrieved (and whether the citation graph re-ranked it), which model wrote it with which prompt version, the guardrail's verdict and its own measurements, the output checks, and the time per step. Stored answers show what the server persisted.
- **Copy and Helpful / Not helpful** on every answer. The rating goes to `POST /feedback` and feeds evaluation, never the learner's mastery.
- **Input.** Enter does not send while an input method is composing (Japanese, Chinese, Korean); the composer stops at the server's 4,000 characters and shows a counter near the limit; it does not auto-focus on touch devices (the keyboard would cover the page).
- **Reading** accepts a pasted arXiv link, checks the paper is in the library before starting, and filters the library list.
- **Theme.** Light, dark, or follow the device (You screen). All colours are tokens in `tokens.css`; a dark theme redefines them. The reading column is centred on wide screens.
- **Accessibility.** Skip link, labelled controls, `aria-live` announcements for answers, errors and stops, visible focus, the History panel is a modal dialog that Escape closes.

The CSP forbids framing the app (`frame-ancestors 'none'`), so responsive checks use the browser's device emulation, not an iframe.
