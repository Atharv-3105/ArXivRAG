"""Influence of each paper inside our corpus: PageRank over the citation graph (ADR 0009).

Why PageRank and not citation count: being cited by influential papers counts more than being
cited by obscure ones, which is exactly "foundational" in a reading list. The graph is small
(thousands of nodes), so a plain power iteration in Python takes milliseconds; it runs after each
ingestion and the scores are stored, never computed per request.

    python -m src.graph.influence        # recompute and store
"""

import asyncio
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


def pagerank(
    nodes: list[str],
    edges: list[tuple[str, str]],
    damping: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict[str, float]:
    """PageRank scaled so the average node scores 1.0 (1.0 = "typical", 3.0 = 3x typical).

    edges are (citing, cited): rank flows from a citing paper to what it cites. Papers citing
    nothing in the corpus ("dangling") spread their rank evenly, the standard fix that keeps the
    total constant.
    """
    n = len(nodes)
    if n == 0:
        return {}
    index = {node: i for i, node in enumerate(nodes)}
    out_links: dict[int, set[int]] = defaultdict(set)
    for src, dst in edges:
        if src in index and dst in index and src != dst:
            out_links[index[src]].add(index[dst])
    incoming: dict[int, list[int]] = defaultdict(list)
    for i, targets in out_links.items():
        for j in targets:
            incoming[j].append(i)

    rank = [1.0 / n] * n
    for _ in range(max_iter):
        dangling = sum(rank[i] for i in range(n) if not out_links.get(i))
        new = [
            (1 - damping) / n
            + damping * (dangling / n + sum(rank[j] / len(out_links[j]) for j in incoming[i]))
            for i in range(n)
        ]
        delta = sum(abs(a - b) for a, b in zip(new, rank, strict=True))
        rank = new
        if delta < tol:
            break
    return {node: round(rank[i] * n, 4) for node, i in index.items()}


def in_degrees(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    degree = dict.fromkeys(nodes, 0)
    for src, dst in set(edges):
        if dst in degree and src != dst:
            degree[dst] += 1
    return degree


async def refresh_influence(store) -> int:
    """Recompute and store influence for the whole corpus. Returns the number of papers scored."""
    nodes, edges = await store.corpus_graph()
    ranks = pagerank(nodes, edges)
    degrees = in_degrees(nodes, edges)
    rows = [{"arxiv_id": n, "in_degree": degrees[n], "pagerank": ranks[n]} for n in nodes]
    await store.save_influence(rows)
    logger.info("Influence refreshed", papers=len(rows), edges=len(edges))
    return len(rows)


def main() -> None:
    from src.core.logging import setup_logging
    from src.graph.store import PgGraphStore
    from src.storage.database import dispose_engine, init_db_schema

    async def run() -> None:
        await init_db_schema()
        try:
            await refresh_influence(PgGraphStore())
        finally:
            await dispose_engine()

    setup_logging()
    asyncio.run(run())


if __name__ == "__main__":
    main()
