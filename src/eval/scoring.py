"""Metric functions for the eval harness (pure, unit-tested)."""

import math
from collections import Counter
from collections.abc import Sequence


def accuracy(pairs: Sequence[tuple[object, object]]) -> float:
    return round(sum(1 for gold, pred in pairs if gold == pred) / len(pairs), 4) if pairs else 0.0


def per_class(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """precision / recall / f1 / support per gold label."""
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    out = {}
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for g, _ in pairs if g == label),
        }
    return out


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    classes = [v for v in per_class(pairs).values() if v["support"]]
    return round(sum(c["f1"] for c in classes) / len(classes), 4) if classes else 0.0


def confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = {}
    for gold, pred in pairs:
        table.setdefault(str(gold), Counter())[str(pred)] += 1
    return {g: dict(c) for g, c in table.items()}


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary relevance; duplicates of one paper count once (first occurrence)."""
    seen: set[str] = set()
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant and item not in seen:
            dcg += 1.0 / math.log2(i + 1)
        seen.add(item)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]
