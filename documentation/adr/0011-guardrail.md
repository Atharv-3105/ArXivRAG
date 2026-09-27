# ADR 0011: Guardrail with an input gate and post-answer grounding checks (flag, don't rewrite)

- **Status:** Accepted (2026-09-27). Plan §4 Guardrail and Phase 6.
- **Code:** `src/agents/guardrail.py`; wired in `src/agents/graph.py` (`_moderation_gate`, `_out_of_domain`, the `guardrail` node)
- **Tests:** `tests/test_guardrail.py`; offline eval suites `domain` and `guardrail` (`src/eval/offline.py`)

## Context

The plan's Guardrail must check citation grounding, out-of-domain queries, toxic content and math consistency, and "block ungrounded claims, suggest fallback, log violations". Two constraints shape how:

1. **Answers are streamed.** By the time an answer can be checked, the reader has seen it. "Block" cannot mean "withhold".
2. **Free tier.** A second LLM call per turn to judge every answer would halve the answers a quota can serve.

Vector search also *always* returns something. Without a gate, "give me a pasta recipe" retrieves the nearest ML chunks, and the LLM improvises from them.

## Decision

**Input (before any retrieval or generation):**
- **Domain gate:** free-form messages with no ML signal (vocabulary concept, arXiv id, research lexicon, or an ML topic in the last 3 user turns) get a fixed reply, with no LLM call. Tutor commands ("quiz me", "hint", "next"), acknowledgements and smalltalk pass. The lexicon is deliberately generous, because a false refusal is worse than a false pass: the grounded pipeline still answers "not in the indexed papers" to anything unrelated it lets through.
- **Prompt-injection markers** are detected, logged and recorded on the answer. The system prompt's grounding rules apply either way, so detection is for monitoring, not blocking.
- **Moderation (toxic or unsafe requests):** an LLM classifier behind `GUARDRAIL_MODERATION`, **off by default** because it costs a call per message. It fails open, since the gated, grounded pipeline is the primary defence and moderation must not take the product down.

**Output (a node after every specialist):**

| Check | Method | Effect |
|---|---|---|
| Ungrounded citation | `[n]` with no retrieved source n | blocks |
| Uncited claims | share of substantive sentences (≥ 6 words, not questions, headings or tables) without `[n]` > 50% | blocks |
| Weak support | a cited sentence shares < 25% of its content words with the cited source | blocks if > half of cited sentences |
| Degraded math quoted | a formula found verbatim in a PDF-extracted (`math_fidelity=degraded`) source | blocks |
| Malformed math | unbalanced braces | recorded only |

**"Blocks" means:** the answer is not approved, and a visible notice is appended to the stream ("Check before relying on this: it cites sources that were not retrieved"). The decision (approved, confidence, issues, metrics) is stored with the message and counted in `arxivmind_guardrail_decisions_total`. Tutor grading and hint replies run in lenient mode: they're not held to citation coverage.

## Alternatives considered

| Option | Why not |
|---|---|
| Buffer the answer, check, then stream | Time to first token becomes the full generation time, which breaks the 2 s p95 target the plan sets. |
| LLM-as-judge on every answer | Doubles LLM calls. Used where it pays: the live eval harness, offline, on samples. |
| NLI model for entailment | A cross-encoder NLI model on CPU per sentence adds hundreds of ms and memory the free host lacks. Lexical support is a cheap first line that catches the blatant cases. The eval can show where NLI would add value. |
| Delete ungrounded sentences after the fact | The reader already saw them. Silent edits between a stream and history are worse than a visible correction. |
| Strict domain classifier (refuse unless clearly ML) | Measured false-refusal risk on real phrasing ("What is Bayes' rule?", "thanks, that helped") before the lexicon was widened. Refusing a real user costs more than letting a stray question reach "not in the indexed papers". |

## Consequences

- Good: no LLM quota is spent on off-topic questions, every answer carries a machine-readable grounding verdict, and the product's grounding rate is a metric with an alert (`GuardrailRejections`).
- Limitation: lexical support can't see paraphrase-level contradiction (a sentence that reuses the source's words but reverses its meaning passes). The offline suite includes a paraphrase that must pass. A contradiction detector is future work, justified only if the live eval shows such cases.
- Limitation: toxicity moderation is opt-in. Without it, only the domain gate and grounding stand between an abusive prompt and a (grounded, on-topic) answer.
