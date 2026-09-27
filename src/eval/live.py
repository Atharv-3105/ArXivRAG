"""Live evaluation: needs the seeded corpus (python -m src.eval.seed) and, for answers, LLM keys.

    python -m src.eval.live retrieval [--write]   # ablation: dense | sparse | hybrid | hybrid+graph
    python -m src.eval.live answers  [--write]    # end-to-end turns: latency, checks, guardrail

Retrieval ablation answers the questions earlier phases left open with numbers: does BM25 add to
dense, does RRF fusion beat either branch, does the graph re-rank help (ADR 0005, 0009)?
Answers measures what the offline mock cannot: real models' trailer / one-question compliance,
guardrail approval rate, and time to first token (the plan's p95 < 2 s target).
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from src.agents.graph_retrieval import rerank
from src.eval import scoring
from src.eval.offline import REPORT_DIR, load
from src.pipeline.citations import base_id

K = 10


async def _ranked(store, graph, query: str, variant: str) -> list[str]:
    mode = "hybrid" if variant.startswith("hybrid") else variant
    hits = await store.search(query, top_k=K * 3, mode=mode)
    if variant == "hybrid+graph" and graph is not None:
        influence = await graph.influence(list(dict.fromkeys(h.paper_id for h in hits)))
        hits = rerank(hits, None, influence)
    return list(dict.fromkeys(base_id(h.paper_id) for h in hits))  # rank papers, not chunks


async def retrieval_ablation() -> dict:
    from src.graph.store import PgGraphStore
    from src.storage.vector_db import VectorStore

    store = VectorStore()
    graph = PgGraphStore()
    rows = load("retrieval.jsonl")
    variants = ["dense", "sparse", "hybrid", "hybrid+graph"]
    table = {}
    try:
        for variant in variants:
            recalls, mrrs, ndcgs = [], [], []
            for r in rows:
                ranked = await _ranked(store, graph, r["query"], variant)
                relevant = set(r["relevant"])
                recalls.append(scoring.recall_at_k(ranked, relevant, 5))
                mrrs.append(scoring.mrr(ranked, relevant))
                ndcgs.append(scoring.ndcg_at_k(ranked, relevant, K))
            n = len(rows)
            table[variant] = {
                "recall@5": round(sum(recalls) / n, 4),
                "mrr": round(sum(mrrs) / n, 4),
                f"ndcg@{K}": round(sum(ndcgs) / n, 4),
            }
    finally:
        await store.close()
    return {"queries": len(rows), "variants": table}


async def answers() -> dict:
    """Runs real turns through the graph with the configured providers (no HTTP, no auth)."""
    from src.api.services import create_services

    svc = await create_services()
    queries = [r["query"] for r in load("routing.jsonl") if r["agent"] != "general"][:20]
    rows = []
    try:
        for q in queries:
            started = time.perf_counter()
            first_token = None
            final: dict = {}
            async for mode, payload in svc.graph.astream(
                {"query": q, "history": [], "timings": {}, "user_id": None},
                stream_mode=["custom", "updates"],
            ):
                if mode == "custom" and payload["type"] == "token" and first_token is None:
                    first_token = time.perf_counter() - started
                elif mode == "updates":
                    for update in payload.values():
                        final.update(update or {})
            rows.append(
                {
                    "query": q,
                    "agent": final.get("agent"),
                    "provider": final.get("provider"),
                    "prompt_version": final.get("prompt_version"),
                    "first_token_s": round(first_token or 0.0, 3),
                    "total_s": round(time.perf_counter() - started, 3),
                    "checks": final.get("checks", {}),
                    "guardrail_approved": (final.get("guardrail") or {}).get("approved"),
                }
            )
    finally:
        await svc.aclose()
    ttft = [r["first_token_s"] for r in rows]
    checks: dict[str, list[bool]] = {}
    for r in rows:
        for name, ok in r["checks"].items():
            checks.setdefault(name, []).append(ok)
    return {
        "turns": len(rows),
        "first_token_p50_s": scoring.percentile(ttft, 50),
        "first_token_p95_s": scoring.percentile(ttft, 95),
        "guardrail_approval_rate": round(
            sum(bool(r["guardrail_approved"]) for r in rows) / max(len(rows), 1), 4
        ),
        "check_pass_rates": {k: round(sum(v) / len(v), 4) for k, v in checks.items()},
        "rows": rows,
    }


def main() -> None:
    cli = argparse.ArgumentParser(description="Live evaluation (needs services / keys)")
    cli.add_argument("suite", choices=["retrieval", "answers"])
    cli.add_argument("--write", action="store_true")
    args = cli.parse_args()
    fn = retrieval_ablation if args.suite == "retrieval" else answers
    result = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), **asyncio.run(fn())}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.write:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        Path(REPORT_DIR / f"live-{args.suite}.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
