# Phase 5: Specialist agents and Socratic pedagogy

> Goal (plan §6 Phase 5 and the §4 agents): Tutor, Critic, Synthesizer and Advisor; prompt templates with versions and A/B tests; one question at a time; hints; confidence tracking; "I don't understand".
> Decisions: [ADR 0010](../adr/0010-specialist-agents.md). Builds on ADR 0008 (mastery) and ADR 0009 (graph).

## 1. The graph now

```
START → route ──(no retrieval needed: grade, hint, skip, smalltalk)──────────┐
          │                                                                  ▼
          └──► retrieve (hybrid+graph | paper section | equation) ──► tutor | critic | synthesizer | advisor | general → END
```

`route` combines the intent (rules, or an optional LLM) with the conversation's tutor state and any explicit UI action. It decides the **agent**, the **tutor action** and a **retrieval plan**.

## 2. Try it (no API key needed: the offline mock follows every contract)

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/register -H 'content-type: application/json' \
  -d '{"email":"me@example.com","password":"correct horse"}' | jq -r .access_token)
curl -s -X POST localhost:8000/api/v1/chat -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"message":"Explain how LoRA works"}' | jq '{agent, action, tutor, prompt_version, checks}'
```

Then continue the same `conversation_id` with: an answer, `hint`, `I don't understand`, `skip`, `quiz me`, `walk me through 2106.09685`, `next`, `explain equation 3 of 2106.09685`. Or send the explicit `action` field (`answer`, `hint`, `dont_understand`, `skip`, `review`, `read`, `next`, `equation`) with `paper_id` / `equation`.

## 3. The specialists

| Agent | When | What it does | Checks |
|---|---|---|---|
| **Tutor** | learning questions, tutor commands | explains one idea at the reader's pitch and asks **one** check question; grades answers; gives hints; re-explains more simply; runs reviews; guided reading; equation walk-throughs | `one_question` / `no_question`, `trailer_parsed`, `citations_grounded` |
| **Critic** | critique intent | peer-review-style assessment in 7 sections, citing every strength and weakness, attributing other papers' views | `all_sections`, `claims_cited` |
| **Synthesizer** | compare / survey intent | comparison table, agreement, disagreement (both sides cited), timeline, open directions | `comparison_table`, `all_sections`, `two_or_more_papers` |
| **Advisor** | "what should I read next?" | deterministic ranking: retrieval rank × weak concepts covered × prerequisite-of × PageRank × interests. Every reason shown. Mentions due reviews. **No LLM call.** | `has_recommendations` |
| **General** | anything else | the Phase 2 grounded answer | `citations_grounded` |

### Tutor session walk-through
1. "Explain how LoRA works" → `explain`: intuition, the idea, one check question. Pending question, answer key (never shown) and compact sources are saved with the conversation.
2. The reader answers → `grade`: the verdict comes from the same streamed call, through the `<<<META>>>` trailer.
   - correct → the next, deeper question (or a wrap-up at `TUTOR_MAX_STEPS`);
   - partial or incorrect → a hint and the same question, and after `TUTOR_MAX_HINTS` an easier explanation (level −1).

   Each verdict is a `quiz` learning event, which moves mastery and the SM-2 schedule.
3. "I don't understand" → `reexplain` at a lower level (analogy, smaller steps), recorded as `confused`.
4. `skip` → closes the question and shows the reference answer (no LLM call).
5. Later, "quiz me" → `review` of the concept whose spaced-repetition review is due.

## 4. Prompt templates
- `src/agents/templates/*.vN.j2`: `_rules.j2` (grounding rules + reader block) and `_trailer.j2` (the META contract) are shared.
- `PROMPT_VARIANTS={"tutor_explain": ["v1","v2"]}`: v1 is intuition first, v2 is example first. Users are split deterministically, and each message stores `prompt_version`, ready for Phase 6 to compare pass rates, grades and feedback per variant.
- Templates are validated at startup (strict variables, every configured variant must exist).

## 5. Failure modes

| Failure | Behaviour |
|---|---|
| Model ignores the trailer | reply shown anyway; the last '?' sentence becomes the pending question; `trailer_parsed: false` recorded |
| Model asks several questions | the last one is pending; `one_question: false` recorded |
| Invalid verdict | no mastery update, session continues |
| Paper / equation not in corpus | fixed message, no LLM call |
| Reading past the last section | "that was the last section", reading mode ends |
| Nothing due for review | fixed message, no LLM call |
| History / state store down | the answer is still produced; the session restarts next turn |

## 6. Acceptance criteria (plan §6, Phase 5)

| Criterion | Status |
|---|---|
| Tutor asks one question at a time | Done: prompt contract plus the `one_question` check on every answer (tests). |
| Adjusts math depth | Done: the reader block (math depth, level) plus pitch from mastery, stepped down by re-explanations. |
| Cites sections | Done: the same grounding as every specialist, with section deep links. |
| Tracks confidence | Done: every graded answer is a quiz event feeding mastery and SM-2. Level and hints are tracked per session. |
| Jinja2 templates, versioning, A/B | Done: A/B logs `prompt_version`. LangSmith receives it as a tag when tracing is on. |
| "I don't understand" → re-explain lower | Done: free text or the explicit `dont_understand` action. |
| Specialist agents from plan §4 | Done: Tutor, Critic, Synthesizer and Advisor, plus General. |
| Live-model behaviour | **Not measured yet.** Tests use the contract-following mock. Real models' `trailer_parsed` and `one_question` rates are a Phase 6 measurement. |

## 7. Interview questions (with the answer you should be able to give)

1. **"Multi-agent or not? How did you decide?"** The tasks are predictable, so specialists are graph nodes with their own retrieval, prompt, contract and checks, at one call per turn. Autonomous tool loops would multiply free-tier quota and latency without a measured gain.
2. **"How do you get structured output *and* token streaming from one call?"** A delimiter trailer: stream the prose, hide everything after `<<<META>>>`, parse its JSON. Handle a delimiter split across tokens by holding back only a possible prefix. Repair if the model ignores the format.
3. **"How does the tutor know an answer from a new question?"** Session state with a pending question. Explicit UI actions first, then commands, then a heuristic (ends in '?', confident other intent, different concepts). The UI buttons make it deterministic.
4. **"Where does the tutoring state live, and why?"** In `conversations.state`, as JSON written in the same transaction as the turn's messages. It's small, queryable, and survives restarts. A checkpointer would add opaque blobs for a three-node graph.
5. **"How do you A/B test prompts in production?"** Versioned template files, a deterministic user split (hash of user id and template), `prompt_version` stored per answer, and outcomes compared per variant: grades, feedback, check pass rates.
6. **"Why don't you retry when an output check fails?"** The text was already streamed. Retrying shows the user two answers. Repair deterministically, record the failure, and fix it at the prompt or model level using the measured rates.
7. **"Why is your recommender not an LLM?"** All the signals are data (gaps, prerequisites, PageRank, interests, retrieval rank). A formula is fast, free and explainable, and its reasons are generated from the same numbers that ranked it.
8. **"How does tutoring feed back into personalisation?"** Each verdict is a quiz event (correct, partial or incorrect) to the async MemoryManager: EMA mastery plus SM-2 scheduling. That drives the next review and how the next explanation is pitched.
9. **"How do you stop the Socratic tutor answer-dumping or over-questioning?"** Put the contract in the prompt, count questions outside math and code in every reply, and track violation rates per template and model. Guided modes (review) explicitly forbid explaining first.
10. **"How do you keep prompts from breaking in production?"** Strict undefined variables, startup validation of every template and configured variant, and a test that renders every template with sample variables.
