# ADR 0010: Specialist agents as graph nodes, a stateful Socratic tutor, and versioned prompts

- **Status:** Accepted (2026-09-27). Covers plan Phase 5 (tutor and pedagogy) and the plan's §4 agents (Tutor, Critic, Synthesizer, Advisor).
- **Code:** `src/agents/graph.py` (route and retrieve), `specialists.py`, `tutor.py`, `advisor.py`, `prompts.py` + `templates/`, `trailer.py`, `checks.py`; `src/storage/paper_content.py`
- **Tests:** `tests/test_agents_units.py`, `tests/test_agents_flow.py`

## Context

The plan names four specialist agents plus a router. On free tiers each LLM call spends scarce quota and adds seconds, and the product's tasks are predictable: explain, critique, compare, recommend. The pedagogy requirements (one question at a time, hints, confidence tracking, "I don't understand", reviews) are **multi-turn**, so they need state.

## Decision

### Specialists are nodes, not autonomous agents
Each specialist is a LangGraph node with its own retrieval strategy, prompt template, output contract and checks, and makes **at most one LLM call per turn**. The graph itself is the orchestrator:

```
route (intent + tutor commands) → retrieve (plan) → tutor | critic | synthesizer | advisor | general
```

| Specialist | Retrieval | Output contract | LLM calls |
|---|---|---|---|
| Tutor | hybrid search + prerequisites for weak readers; stored sources when grading; a paper section (reading); an equation plus its section | a reply ending in exactly one question, plus a JSON trailer | 1 (0 for skip or nothing due) |
| Critic | hybrid search + what other papers say (ADR 0009) | 7 fixed sections; every strength and weakness cited | 1 |
| Synthesizer | broad: 8 sources, at most 2 per paper, plus citation-connected papers | table, agree/disagree, timeline, open directions; at least 2 papers | 1 |
| Advisor | candidates by topic (or by the reader's gaps when the question has none) | ranked list with computed reasons | **0**: deterministic ranking |
| General | hybrid search (none for smalltalk) | grounded answer | 1 |

### The tutor is a state machine
Session state is saved with the conversation (`conversations.state`, written in the same transaction as the turn's messages): mode, topic, concept, step, level, hints used, the pending question with its answer key, compact source copies for grading, and the reading or equation position.

Actions: `explain`, `grade`, `hint`, `reexplain`, `skip`, `review`, `read`, `read_next`, `equation`.

- **Detection:** explicit UI actions come first; free-text commands next ("hint", "I don't understand", "next", "quiz me", "walk me through <id>", "equation 3 of <id>"). While a question is pending, a message is treated as an answer unless it is clearly a new question (ends in "?", confident non-chat intent, different concepts).
- **Grading** happens in the same single call as the reply. The prompt asks for prose, then `<<<META>>>`, then one line of JSON (`verdict`, next `question`, `answer_key`). A streaming splitter shows everything before the delimiter and hides the rest, holding back only the characters that could start a delimiter split across two tokens.
- **Learning signals:** a verdict becomes a `quiz` event (correct 1.0, partial 0.5, incorrect 0.0), the strongest evidence the mastery model gets (ADR 0008). "I don't understand" becomes a `confused` event and steps the explanation level down (floor −2). Running out of hints also steps down.
- **Modes:**
  - review asks a question on a concept due for spaced repetition, without explaining first;
  - guided reading walks a paper section by section in parser order, starting with its prerequisites;
  - equation walk-through goes symbol by symbol, using the stored LaTeX, never a re-typed copy.

### Prompts are versioned files
- `templates/<name>.v<N>.j2` (Jinja2, `StrictUndefined`). Every answer stores `name@vN` on its message.
- A/B: `PROMPT_VARIANTS` lists the versions in rotation. Assignment is by `sha256(user_id:name)`, so each user always sees the same variant. `tutor_explain` ships v1 (intuition first) and v2 (example first).
- All templates are compiled at startup, and every configured variant must exist, so a broken template stops the deploy, not a user's turn.

### Checks repair or measure; they do not retry
Checks run on every answer and are stored on it:
- tutor: exactly one question (or none on the final wrap-up), trailer parsed;
- critic: all sections present, claims cited;
- synthesizer: table present, all sections, at least 2 papers;
- everyone: citations grounded.

Streamed text has already been read, and regenerating would show one answer and then replace it. So a failure is **repaired** deterministically when possible (no trailer → the last '?' sentence becomes the pending question) and otherwise **measured**: the Phase 6 eval reports violation rates per template version and model. This replaces the "retry once" idea from planning, which only works without streaming.

## Alternatives considered

| Option | Why not |
|---|---|
| Autonomous tool-calling agents (ReAct) per specialist | 2–4× the calls and latency on free quotas, and non-deterministic flow. Our tasks don't need open-ended planning. |
| An LLM supervisor/planner choosing the specialist | One extra call per turn, where rules and the tutor state already decide. |
| A second call for grading, or JSON mode | JSON mode breaks streaming, and a second call doubles cost. The trailer does both in one. |
| Tutor state inside LangGraph's checkpointer | Opaque blobs (ADR 0007). The session is small, queryable JSON next to the messages. |
| Advisor ranking by LLM | Slower, costlier and unexplainable. Every signal it would use is already data we hold. |
| Prompts as Python strings | No versioning, no A/B, and diffs mix prompt edits with code edits. |

## Consequences

- Good: every turn costs at most one call, whichever specialist answers. The tutor's behaviour is testable as a pure state machine, and the offline mock follows every contract, so the full product works without API keys.
- Good: each answer is attributable to a specialist, a template version and a model, with its check results. That's what Phase 6 needs to compare variants.
- Cost: the trailer relies on the model following the format. The repair path and the `trailer_parsed` rate make that visible, and it's the first thing to check when a new provider is added.
- Limitation: the answer/new-question heuristic can misroute. The UI's explicit actions (answer, hint, "I don't understand", next) remove the guessing.
