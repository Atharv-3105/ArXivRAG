# Offline evaluation report

Generated 2026-09-27T05:12:23+00:00 by `python -m src.eval.offline --write`.

| Suite | Metrics |
|---|---|
| routing | agent_accuracy=0.9778, n=45 |
| domain | false_refusal_rate=0.0, off_topic_recall=1.0, n=40 |
| tutor | action_accuracy=1.0, n=33 |
| citation_function | macro_f1=0.8542, accuracy=0.85, n=40 |
| guardrail | decision_accuracy=1.0, n=10 |
| templates | valid=1.0, templates=10 |

## Gate

- all thresholds met

## routing: failures (1)

- `{"query": "What are the prerequisites for understanding Mamba?", "expected": "advisor", "got": "tutor"}`

## citation_function: failures (6)

- `{"sentence": "As noted by [5], this estimator is biased for small samples.", "expected": "critical", "got": "neutral"}`
- `{"sentence": "Unfortunately, [11] reports results on a single seed only.", "expected": "critical", "got": "neutral"}`
- `{"sentence": "However, we use a larger batch size than [2].", "expected": "comparative", "got": "builds_on"}`
- `{"sentence": "While [5] is elegant, its memory cost grows quadratically.", "expected": "critical", "got": "neutral"}`
- `{"sentence": "We note that [9] evaluates only on English.", "expected": "critical", "got": "neutral"}`
- `{"sentence": "Recent work questions the robustness of these results [10].", "expected": "critical", "got": "neutral"}`
