# ADR 0014: The web app is a React SPA in the Margin design, served by the API

- **Status:** Accepted (2026-09-27). The plan's client layer (§3: "Web UI (Next.js/Streamlit)").
- **Code:** `frontend/` (React 19 + TypeScript + Vite), `src/api/frontend.py` (serving), `src/api/demo.py` (offline demo); new endpoints `GET /papers/map`, `GET /me/papers`, `GET /conversations/{id}/tutor`
- **Tests:** `frontend/src/lib/render.test.ts` (Vitest), `tests/test_web_app.py`
- **Design:** three directions were compared on a design canvas. The chosen one is **Margin** (reading-first), with the citation-map view from **Atlas**.

## Context

ArXivMind's job is understanding papers with trustworthy citations, taught by a tutor that asks back. The frontend must stream answers, render LaTeX, show where every claim comes from, run the tutor's multi-turn loop, and show progress. It must fit the free tier (one small container) and not weaken the security posture.

## Decision

### Design: Margin, plus Atlas for the map
- **Answers are typeset pages** (STIX Two Text, a maths-grade serif). Each paragraph's sources sit as **sidenotes beside it**, so a reader checks a claim without leaving the sentence. Mechanically: the answer is split into markdown blocks, and each block lists the citations it introduces first.
- The tutor's open question is an **Exercise** block (textbook convention), with Check my answer, Hint, I don't understand and Skip. Answers are graded with a verdict chip. Reading mode shows "Section i of n" with Next section.
- **Paper map (Atlas):** papers are places, arXiv categories are regions, and influence is elevation. Contours come from a kernel density estimate weighted by PageRank. Filled markers are papers your answers have cited, and dashed trails are "builds on" or "criticises" edges. Labels are placed like a map-maker's: the most influential first, each on the first free side, avoiding other labels and markers. A sheet shows prerequisites and what other papers say, with "Read it with the tutor".
- **One accent** (ink blue), cool paper and hairline rules only where they carry structure. The map uses Atlas's cartographic palette and faces (IBM Plex Sans Condensed, Newsreader).

### Architecture
| Choice | Reason |
|---|---|
| **SPA (React + TypeScript + Vite), no SSR** | Every screen is behind sign-in and is interactive; SSR buys nothing here. |
| **Served by FastAPI from `frontend/dist`** | One origin: no CORS and no second free-tier service. The Docker image builds the frontend in a Node stage, and only static files ship. |
| **SSE read from `fetch`** | `EventSource` cannot POST or send a bearer token. A small parser reads the stream, and tokens are flushed once per animation frame. |
| **markdown-it (`html: false`) + KaTeX** | Model output can never inject HTML. Maths is protected behind placeholders before markdown and restored after citations are linked, so a bracket inside a formula is never mistaken for a citation. |
| **Token in localStorage + a strict CSP** | Survives reloads. The injection risk is bounded by `default-src 'self'`, `script-src 'self'` and `frame-ancestors 'none'` on the app shell. |
| **Fonts self-hosted (Fontsource), map lazy-loaded** | No third-party requests, which the CSP forbids anyway. d3 is only downloaded when the map opens. |
| **An offline demo server** (`python -m src.api.demo`) | The whole product runs with in-memory stores, a built-in corpus and the mock LLM, for development, screenshots and demos without Docker or keys. |

## Alternatives considered

| Option | Why not |
|---|---|
| Next.js | A Node server beside the API, a second service on the free tier, for rendering nobody benefits from. |
| Streamlit | Fast to build, but it can't do sidenote layout, custom streaming, or an accessible tutor flow. It's a notebook, not a product. |
| Deploying the SPA separately (e.g. a static host) | Cross-origin calls need CORS and put a token-bearing API on another origin, for no gain at this scale. It remains an easy change later: the SPA calls relative `/api` paths. |
| A d3 force graph as the paper view | The Phase 4 viewer did this. A generic node-link graph says "graph", not "where you are in the field". The terrain adds meaning (fields, influence, what you've studied). |

## Consequences

- Good: the answer screen makes grounding visible (a sidenote per claim) and the tutor loop explicit, and it works at phone width (the rail becomes a bottom bar and sidenotes fold under each paragraph).
- Good: verified in the running demo, including sign-up, tutoring with grading, guided reading, critique, comparison, advisor, off-topic refusal, reviews, profile and the map. Label collisions on the map were measured (0 overlaps) rather than eyeballed.
- Cost: the main bundle is about 210 KB gzipped (React, KaTeX, markdown-it). The map is split out at about 28 KB gzipped.
- Not done: no end-to-end browser tests in CI yet (Playwright), and no visual regression screenshots. Both are natural next steps.
