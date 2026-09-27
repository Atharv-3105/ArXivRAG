# Evaluation

Methodology: [ADR 0012](../adr/0012-evaluation-and-observability.md). The latest generated offline numbers are in [offline-report.md](offline-report.md) (regenerate with `python -m src.eval.offline --write`).

## What is measured, and what is not yet

| Question | How | Status (2026-09-27) |
|---|---|---|
| Does the router send questions to the right specialist? | offline `routing` (45 labelled queries) | **97.8%**. 1 known miss ("prerequisites for understanding X" goes to the tutor, not the advisor), left unfixed on purpose |
| Does the domain gate refuse real ML questions? | offline `domain` (40) + routing | **0%** false refusals, **100%** off-topic recall |
| Does the tutor tell answers, commands and new questions apart? | offline `tutor` (33) | **100%** |
| Citation function (critical / comparative / builds_on / neutral) | offline `citation_function` (40) | macro-F1 **0.854**. Misses are *implicit* criticism ("evaluates only on English", "while X is elegant…") |
| Guardrail decisions | offline `guardrail` (10 answer/source pairs) | **100%** |
| App overhead per turn (LLM faked) | `python -m loadtest.bench` | p50 **22 ms**, p95 **26 ms**, about 45 turns/s per process |
| Does BM25 / RRF / the graph improve retrieval? | `python -m src.eval.live retrieval` on the seeded corpus | **not run**: needs Qdrant (Docker was down) |
| Real models: trailer, one-question compliance, grounding, first-token p95 | `python -m src.eval.live answers` | **not run**: needs the corpus and LLM keys |
| Prompt A/B (tutor_explain v1 vs v2) | `python -m src.eval.ab` | **no data** until real users |
| Image size, RSS without PyTorch | CI "Report image size" step | **not run**: no remote or CI run yet |

## What the evaluation changed

The first run of the offline suites found three real defects. Each was fixed at its cause, not by adding the failing sentence as a special case:
1. **Routing ties:** generic question openers ("what are", "how do") counted as full evidence for *learn*. So "What are the weaknesses of X?" tied with *critique* and fell back to chat. The fix was to weight the evidence (generic openers 0.5, comparison cues 1.5), which took routing from 82.2% to 97.8%.
2. **Plural concepts:** "VAEs" didn't match the concept "vae", so "What does the reparameterization trick in VAEs do?" was **refused** by the domain gate. The fix was an optional plural suffix in the concept matcher.
3. **Domain gate gaps:** acknowledgements ("thanks, that helped") and statistics vocabulary ("Bayes' rule") were refused. The fix was an acknowledgement pattern and a wider lexicon.

Honesty note: the routing and domain sets both *found* these defects and *measure* the fixes. The numbers after a fix are therefore optimistic. A held-out set written by someone else is the fix for that, and it's a good first contribution.

## Running the live suites

```bash
docker compose up -d db qdrant
python -m src.eval.seed                       # 29 well-known papers by arXiv id
python -m src.eval.live retrieval --write     # -> documentation/eval/live-retrieval.json
python -m src.eval.live answers --write       # needs at least one LLM key in .env
```
