"""Offline evaluation suites: deterministic, no network, run in CI as a gate (ADR 0012).

    python -m src.eval.offline            # print the report, exit 1 if a threshold fails
    python -m src.eval.offline --write    # also write documentation/eval/offline-report.{md,json}

Each suite scores one decision the product makes with rules instead of an LLM, against labelled
data in src/eval/data. The labels were written from realistic phrasing before the suites were
first run, and the failures are listed in the report rather than hidden: a regression suite that
only contains cases the rules already pass measures nothing.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.agents import guardrail, intent, prompts, tutor
from src.agents.graph import AGENT_FOR_INTENT
from src.eval import scoring
from src.memory import concepts as vocab
from src.pipeline.citations import citation_function

DATA = Path(__file__).resolve().parent / "data"
REPORT_DIR = Path(__file__).resolve().parents[2] / "documentation" / "eval"


def load(name: str) -> list[dict]:
    with open(DATA / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def route_fresh(query: str) -> str:
    """The agent a first message of a new conversation reaches (the route node, no state)."""
    concepts = vocab.extract_concepts(query)
    result = intent.classify_rules(query)
    action, _ = tutor.detect(query, None, None, None, result, concepts)
    if action:
        return "tutor"
    if not guardrail.in_domain(query, [], None, None):
        return "refused"
    return AGENT_FOR_INTENT[result.intent]


def suite_routing() -> dict:
    rows = load("routing.jsonl")
    pairs = [(r["agent"], route_fresh(r["query"])) for r in rows]
    return {
        "metrics": {"agent_accuracy": scoring.accuracy(pairs), "n": len(pairs)},
        "per_class": scoring.per_class(pairs),
        "confusion": scoring.confusion(pairs),
        "failures": [
            {"query": r["query"], "expected": g, "got": p}
            for r, (g, p) in zip(rows, pairs, strict=True)
            if g != p
        ],
    }


def suite_domain() -> dict:
    rows = load("domain.jsonl")
    preds = [guardrail.in_domain(r["query"], [], None, None) for r in rows]
    in_rows = [(r, p) for r, p in zip(rows, preds, strict=True) if r["in_domain"]]
    out_rows = [(r, p) for r, p in zip(rows, preds, strict=True) if not r["in_domain"]]
    false_refusals = [r["query"] for r, p in in_rows if not p]
    missed = [r["query"] for r, p in out_rows if p]
    return {
        "metrics": {
            "false_refusal_rate": round(len(false_refusals) / len(in_rows), 4),
            "off_topic_recall": round(1 - len(missed) / len(out_rows), 4),
            "n": len(rows),
        },
        "failures": [{"query": q, "error": "refused an ML question"} for q in false_refusals]
        + [{"query": q, "error": "let an off-topic question through"} for q in missed],
    }


_STATES = {
    "none": None,
    "pending": {"concept": "lora", "pending": {"question": "Why low rank?", "answer_key": "k"}},
    "session": {"concept": "lora", "topic": "LoRA"},
    "reading": {"reading": {"paper_id": "2106.09685", "index": 0}},
}


def suite_tutor() -> dict:
    rows = load("tutor_actions.jsonl")
    pairs = []
    for r in rows:
        result = intent.classify_rules(r["message"])
        if r.get("intent"):  # the labelled intent the classifier should have produced
            result = intent.IntentResult(r["intent"], 1.0, "label")
        action, _ = tutor.detect(
            r["message"], _STATES[r["state"]], None, None, result, r.get("concepts", [])
        )
        pairs.append((str(r["expected"]), str(action)))
    return {
        "metrics": {"action_accuracy": scoring.accuracy(pairs), "n": len(pairs)},
        "failures": [
            {"message": r["message"], "state": r["state"], "expected": g, "got": p}
            for r, (g, p) in zip(rows, pairs, strict=True)
            if g != p
        ],
    }


def suite_citation_function() -> dict:
    rows = load("citation_function.jsonl")
    pairs = [(r["label"], citation_function(r["sentence"])) for r in rows]
    return {
        "metrics": {
            "macro_f1": scoring.macro_f1(pairs),
            "accuracy": scoring.accuracy(pairs),
            "n": len(pairs),
        },
        "per_class": scoring.per_class(pairs),
        "failures": [
            {"sentence": r["sentence"], "expected": g, "got": p}
            for r, (g, p) in zip(rows, pairs, strict=True)
            if g != p
        ],
    }


def suite_guardrail() -> dict:
    rows = load("guardrail.jsonl")
    results = []
    for r in rows:
        decision = guardrail.check_answer(r["answer"], r["sources"], r["ungrounded"])
        issue_ok = r.get("issue") is None or r["issue"] in decision.issues
        results.append((r, decision, decision.approved == r["approved"] and issue_ok))
    return {
        "metrics": {
            "decision_accuracy": round(sum(ok for *_, ok in results) / len(results), 4),
            "n": len(results),
        },
        "failures": [
            {
                "id": r["id"],
                "expected_approved": r["approved"],
                "expected_issue": r.get("issue"),
                "got_approved": d.approved,
                "got_issues": d.issues,
            }
            for r, d, ok in results
            if not ok
        ],
    }


def suite_templates() -> dict:
    try:
        versions = prompts.validate_templates()
        return {"metrics": {"valid": 1.0, "templates": len(versions)}, "failures": []}
    except Exception as e:
        return {"metrics": {"valid": 0.0}, "failures": [{"error": str(e)}]}


SUITES = {
    "routing": suite_routing,
    "domain": suite_domain,
    "tutor": suite_tutor,
    "citation_function": suite_citation_function,
    "guardrail": suite_guardrail,
    "templates": suite_templates,
}


def gate(results: dict) -> list[str]:
    """Threshold violations. Keys ending in _max are upper bounds."""
    thresholds = json.loads((DATA / "thresholds.json").read_text(encoding="utf-8"))
    violations = []
    for key, bound in thresholds.items():
        if key.startswith("_"):
            continue
        suite, metric = key.split(".", 1)
        upper = metric.endswith("_max")
        value = results[suite]["metrics"][metric.removesuffix("_max")]
        if (value > bound) if upper else (value < bound):
            violations.append(f"{key}: {value} {'>' if upper else '<'} {bound}")
    return violations


def run() -> dict:
    results = {name: fn() for name, fn in SUITES.items()}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "results": results,
        "violations": gate(results),
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# Offline evaluation report",
        "",
        f"Generated {report['generated_at']} by `python -m src.eval.offline --write`.",
        "",
        "| Suite | Metrics |",
        "|---|---|",
    ]
    for name, res in report["results"].items():
        shown = ", ".join(f"{k}={v}" for k, v in res["metrics"].items())
        lines.append(f"| {name} | {shown} |")
    lines += ["", "## Gate", ""]
    lines += [f"- FAIL {v}" for v in report["violations"]] or ["- all thresholds met"]
    for name, res in report["results"].items():
        if res.get("failures"):
            lines += ["", f"## {name}: failures ({len(res['failures'])})", ""]
            lines += [f"- `{json.dumps(f, ensure_ascii=False)}`" for f in res["failures"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    cli = argparse.ArgumentParser(description="Offline evaluation suites (CI gate)")
    cli.add_argument("--write", action="store_true", help="write documentation/eval/ reports")
    args = cli.parse_args()
    report = run()
    print(to_markdown(report))
    if args.write:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "offline-report.md").write_text(to_markdown(report), encoding="utf-8")
        (REPORT_DIR / "offline-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    sys.exit(1 if report["violations"] else 0)


if __name__ == "__main__":
    main()
