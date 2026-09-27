"""Citation-graph queries over Postgres (ADR 0003, ADR 0009).

Edges are `paper_citations(citing_id -> cited_base_id)`; a cited paper is "in the corpus" when some
`papers.base_id` matches. Every query runs in a transaction with `SET LOCAL statement_timeout`, so a
pathological traversal is cancelled by the database instead of pinning a connection (plan §4:
"query timeout limits"); traversal depth and result sizes are capped as well.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Protocol

from sqlalchemy import delete, text
from sqlalchemy.dialects.postgresql import insert

from src.core.config import settings
from src.pipeline.citations import base_id
from src.storage.database import PaperInfluence, get_async_session

# Order of interest when one edge has several functions (a paper can cite another twice).
FUNCTION_PRIORITY = ("critical", "comparative", "builds_on", "neutral")


def _strongest(functions) -> str:
    present = set(functions or [])
    return next((f for f in FUNCTION_PRIORITY if f in present), "neutral")


class GraphStore(Protocol):
    async def corpus_graph(self) -> tuple[list[str], list[tuple[str, str]]]: ...

    async def save_influence(self, rows: list[dict]) -> None: ...

    async def influence(self, arxiv_ids: list[str]) -> dict[str, float]: ...

    async def paper(self, arxiv_id: str) -> dict | None: ...

    async def prerequisites(self, arxiv_id: str, max_depth: int, limit: int) -> list[dict]: ...

    async def citing_contexts(
        self, arxiv_id: str, functions: list[str] | None, limit: int
    ) -> list[dict]: ...

    async def neighbors(self, arxiv_ids: list[str], limit: int) -> list[str]: ...

    async def neighborhood(self, arxiv_id: str, depth: int, max_nodes: int) -> dict: ...

    async def top_influential(self, limit: int) -> list[dict]: ...

    async def missing(self, limit: int) -> list[dict]: ...

    async def map(self, limit: int) -> dict: ...


# ================================================================== Postgres

_PAPER = text(
    """
    SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.published_date, p.categories,
           p.parse_method, p.base_id,
           COALESCE(i.in_degree, 0) AS in_degree, COALESCE(i.pagerank, 0) AS pagerank,
           (SELECT count(*) FROM paper_citations c
             WHERE c.citing_id = p.arxiv_id) AS reference_count,
           (SELECT count(*) FROM paper_citations c WHERE c.cited_base_id = p.base_id) AS cited_by
    FROM papers p LEFT JOIN paper_influence i ON i.arxiv_id = p.arxiv_id
    WHERE p.arxiv_id = :id OR p.base_id = :id
    ORDER BY p.arxiv_id DESC
    LIMIT 1
    """
)

# Papers `root` builds on, transitively, inside the corpus. The path array is the cycle guard
# (citation graphs are not DAGs: versions and simultaneous papers cite each other).
_PREREQUISITES = text(
    """
    WITH RECURSIVE chain(arxiv_id, depth, path) AS (
        SELECT p.arxiv_id, 1, ARRAY[CAST(:root AS VARCHAR), p.arxiv_id]
        FROM paper_citations c JOIN papers p ON p.base_id = c.cited_base_id
        WHERE c.citing_id = :root AND p.arxiv_id <> :root
      UNION ALL
        SELECT p.arxiv_id, ch.depth + 1, ch.path || p.arxiv_id
        FROM chain ch
        JOIN paper_citations c ON c.citing_id = ch.arxiv_id
        JOIN papers p ON p.base_id = c.cited_base_id
        WHERE ch.depth < :max_depth AND NOT (p.arxiv_id = ANY(ch.path))
    ),
    best AS (
        SELECT DISTINCT ON (arxiv_id) arxiv_id, depth, path FROM chain ORDER BY arxiv_id, depth
    )
    SELECT b.arxiv_id, b.depth, b.path, p.title, p.published_date,
           COALESCE(i.pagerank, 0) AS pagerank
    FROM best b JOIN papers p ON p.arxiv_id = b.arxiv_id
    LEFT JOIN paper_influence i ON i.arxiv_id = b.arxiv_id
    ORDER BY b.depth, pagerank DESC, p.published_date
    LIMIT :limit
    """
)

_CONTEXTS = text(
    """
    SELECT x.citing_id, p.title AS citing_title, x.sentence, x.function, x.anchor
    FROM citation_contexts x JOIN papers p ON p.arxiv_id = x.citing_id
    WHERE x.cited_base_id = :base
      AND (CAST(:functions AS VARCHAR[]) IS NULL OR x.function = ANY(CAST(:functions AS VARCHAR[])))
    ORDER BY array_position(ARRAY['critical','comparative','builds_on','neutral'], x.function),
             p.published_date DESC
    LIMIT :limit
    """
)

_NEIGHBORS = text(
    """
    SELECT DISTINCT other FROM (
        SELECT p.arxiv_id AS other
        FROM paper_citations c JOIN papers p ON p.base_id = c.cited_base_id
        WHERE c.citing_id = ANY(:ids)
        UNION
        SELECT c.citing_id AS other FROM paper_citations c
        WHERE c.cited_base_id = ANY(:bases)
    ) n
    WHERE NOT (other = ANY(:ids))
    LIMIT :limit
    """
)

_OUT_EDGES = text(
    """
    SELECT c.citing_id AS src, c.cited_base_id, p.arxiv_id AS dst, p.title, p.published_date
    FROM paper_citations c LEFT JOIN papers p ON p.base_id = c.cited_base_id
    WHERE c.citing_id = ANY(:ids)
    """
)

_IN_EDGES = text(
    """
    SELECT c.citing_id AS src, p.title, p.published_date, c.cited_base_id
    FROM paper_citations c JOIN papers p ON p.arxiv_id = c.citing_id
    WHERE c.cited_base_id = ANY(:bases)
    """
)

_EDGE_FUNCTIONS = text(
    """
    SELECT citing_id, cited_base_id, array_agg(DISTINCT function) AS functions
    FROM citation_contexts
    WHERE citing_id = ANY(:ids) OR cited_base_id = ANY(:bases)
    GROUP BY citing_id, cited_base_id
    """
)

_CORPUS_EDGES = text(
    """
    SELECT c.citing_id, p.arxiv_id
    FROM paper_citations c JOIN papers p ON p.base_id = c.cited_base_id
    """
)

_TOP = text(
    """
    SELECT p.arxiv_id, p.title, p.published_date, i.in_degree, i.pagerank
    FROM paper_influence i JOIN papers p ON p.arxiv_id = i.arxiv_id
    ORDER BY i.pagerank DESC LIMIT :limit
    """
)

# The most-cited papers we do not have: a ready-made "ingest next" list.
_MISSING = text(
    """
    SELECT c.cited_base_id AS arxiv_id, count(DISTINCT c.citing_id) AS cited_by
    FROM paper_citations c
    WHERE NOT EXISTS (SELECT 1 FROM papers p WHERE p.base_id = c.cited_base_id)
    GROUP BY c.cited_base_id
    ORDER BY cited_by DESC, c.cited_base_id
    LIMIT :limit
    """
)


# The map: the most influential papers, and the citation edges among them.
_MAP_NODES = text(
    """
    SELECT p.arxiv_id, p.title, p.published_date, p.categories->>0 AS category,
           COALESCE(i.pagerank, 0) AS pagerank, COALESCE(i.in_degree, 0) AS in_degree
    FROM papers p LEFT JOIN paper_influence i ON i.arxiv_id = p.arxiv_id
    WHERE p.index_status = 'indexed'
    ORDER BY pagerank DESC, p.published_date DESC
    LIMIT :limit
    """
)
_MAP_EDGES = text(
    """
    SELECT c.citing_id AS source, p.arxiv_id AS target,
           (SELECT array_agg(DISTINCT x.function) FROM citation_contexts x
             WHERE x.citing_id = c.citing_id AND x.cited_base_id = c.cited_base_id) AS functions
    FROM paper_citations c JOIN papers p ON p.base_id = c.cited_base_id
    WHERE c.citing_id = ANY(:ids) AND p.arxiv_id = ANY(:ids) AND p.arxiv_id <> c.citing_id
    """
)


def _iso(value) -> str | None:
    return value.date().isoformat() if isinstance(value, datetime) else None


class PgGraphStore:
    def __init__(self, timeout_ms: int | None = None):
        self.timeout_ms = int(timeout_ms or settings.GRAPH_TIMEOUT_SECONDS * 1000)

    @asynccontextmanager
    async def _tx(self, timeout_ms: int | None = None) -> AsyncIterator:
        async with get_async_session() as session:
            async with session.begin():
                # An int we control, so formatting is safe (SET cannot take bind parameters).
                ms = int(timeout_ms or self.timeout_ms)
                await session.execute(text(f"SET LOCAL statement_timeout = {ms}"))
                yield session

    async def corpus_graph(self):
        async with self._tx(timeout_ms=60_000) as s:  # offline job: generous timeout
            nodes = list((await s.execute(text("SELECT arxiv_id FROM papers"))).scalars())
            edges = [(r[0], r[1]) for r in (await s.execute(_CORPUS_EDGES)).all()]
        return nodes, edges

    async def save_influence(self, rows):
        async with self._tx(timeout_ms=60_000) as s:
            await s.execute(delete(PaperInfluence))  # full recompute replaces the whole table
            if rows:
                await s.execute(insert(PaperInfluence).values(rows))

    async def influence(self, arxiv_ids):
        if not arxiv_ids:
            return {}
        async with self._tx() as s:
            rows = await s.execute(
                text("SELECT arxiv_id, pagerank FROM paper_influence WHERE arxiv_id = ANY(:ids)"),
                {"ids": arxiv_ids},
            )
            return {r.arxiv_id: r.pagerank for r in rows}

    async def paper(self, arxiv_id):
        async with self._tx() as s:
            row = (await s.execute(_PAPER, {"id": arxiv_id})).mappings().one_or_none()
        if row is None:
            return None
        data = dict(row)
        data["published"] = _iso(data.pop("published_date"))
        return data

    async def prerequisites(self, arxiv_id, max_depth, limit):
        async with self._tx() as s:
            rows = await s.execute(
                _PREREQUISITES,
                {"root": arxiv_id, "max_depth": min(max_depth, 4), "limit": limit},
            )
            return [
                {
                    "arxiv_id": r.arxiv_id,
                    "title": r.title,
                    "depth": r.depth,
                    "path": list(r.path),
                    "pagerank": r.pagerank,
                    "published": _iso(r.published_date),
                }
                for r in rows
            ]

    async def citing_contexts(self, arxiv_id, functions, limit):
        async with self._tx() as s:
            rows = await s.execute(
                _CONTEXTS, {"base": base_id(arxiv_id), "functions": functions, "limit": limit}
            )
            return [dict(r._mapping) for r in rows]

    async def neighbors(self, arxiv_ids, limit):
        if not arxiv_ids:
            return []
        async with self._tx() as s:
            rows = await s.execute(
                _NEIGHBORS,
                {"ids": arxiv_ids, "bases": [base_id(i) for i in arxiv_ids], "limit": limit},
            )
            return list(rows.scalars())

    async def neighborhood(self, arxiv_id, depth, max_nodes):
        """BFS in both directions, one query pair per hop, capped at max_nodes (payload limit)."""
        nodes: dict[str, dict] = {}
        edges: dict[tuple[str, str], dict] = {}
        root = await self.paper(arxiv_id)
        if root is None:
            return {"nodes": [], "edges": [], "truncated": False}
        nodes[root["arxiv_id"]] = {
            "id": root["arxiv_id"],
            "title": root["title"],
            "published": root["published"],
            "in_corpus": True,
            "depth": 0,
        }
        frontier, truncated = [root["arxiv_id"]], False
        async with self._tx() as s:
            for hop in range(1, min(depth, 3) + 1):
                if not frontier or truncated:
                    break
                bases = [base_id(i) for i in frontier]
                found: list[str] = []
                for r in (await s.execute(_OUT_EDGES, {"ids": frontier})).all():
                    dst = r.dst or r.cited_base_id  # external papers are nodes too
                    truncated |= self._add(
                        nodes, dst, r.title, r.published_date, r.dst, hop, max_nodes, found
                    )
                    if dst in nodes:
                        edges[(r.src, dst)] = {
                            "source": r.src,
                            "target": dst,
                            "function": "neutral",
                        }
                for r in (await s.execute(_IN_EDGES, {"bases": bases})).all():
                    target = next((f for f in frontier if base_id(f) == r.cited_base_id), None)
                    truncated |= self._add(
                        nodes, r.src, r.title, r.published_date, r.src, hop, max_nodes, found
                    )
                    if target and r.src in nodes:
                        edges[(r.src, target)] = {
                            "source": r.src,
                            "target": target,
                            "function": "neutral",
                        }
                frontier = [f for f in found if nodes[f]["in_corpus"]]

            ids = [n for n, v in nodes.items() if v["in_corpus"]]
            for r in (
                await s.execute(_EDGE_FUNCTIONS, {"ids": ids, "bases": [base_id(i) for i in ids]})
            ).all():
                for key, edge in edges.items():
                    if key[0] == r.citing_id and base_id(key[1]) == r.cited_base_id:
                        edge["function"] = _strongest(r.functions)
        influence = await self.influence([n for n, v in nodes.items() if v["in_corpus"]])
        for node in nodes.values():
            node["pagerank"] = influence.get(node["id"], 0.0)
        return {
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "truncated": truncated,
        }

    @staticmethod
    def _add(nodes, node_id, title, published, corpus_id, hop, max_nodes, found) -> bool:
        """Add a node unless present; True if it had to be dropped because of the cap."""
        if node_id in nodes:
            return False
        if len(nodes) >= max_nodes:
            return True
        nodes[node_id] = {
            "id": node_id,
            "title": title,
            "published": _iso(published),
            "in_corpus": corpus_id is not None,
            "depth": hop,
        }
        found.append(node_id)
        return False

    async def top_influential(self, limit):
        async with self._tx() as s:
            rows = await s.execute(_TOP, {"limit": limit})
            return [
                {
                    "arxiv_id": r.arxiv_id,
                    "title": r.title,
                    "published": _iso(r.published_date),
                    "in_degree": r.in_degree,
                    "pagerank": r.pagerank,
                }
                for r in rows
            ]

    async def map(self, limit):
        async with self._tx() as s:
            nodes = [
                {
                    "id": r.arxiv_id,
                    "title": r.title,
                    "published": _iso(r.published_date),
                    "category": r.category,
                    "pagerank": r.pagerank,
                    "in_degree": r.in_degree,
                }
                for r in await s.execute(_MAP_NODES, {"limit": limit})
            ]
            ids = [n["id"] for n in nodes]
            edges = []
            if ids:
                rows = await s.execute(_MAP_EDGES, {"ids": ids})
                edges = [
                    {"source": r.source, "target": r.target, "function": _strongest(r.functions)}
                    for r in rows
                ]
        return {"nodes": nodes, "edges": edges}

    async def missing(self, limit):
        async with self._tx() as s:
            rows = await s.execute(_MISSING, {"limit": limit})
            return [{"arxiv_id": r.arxiv_id, "cited_by": r.cited_by} for r in rows]


# ================================================================== in memory (tests)


class InMemoryGraphStore:
    """The same queries over Python structures. `papers`: arxiv_id -> {title, published}."""

    def __init__(self) -> None:
        self.papers: dict[str, dict] = {}
        self.edges: set[tuple[str, str]] = set()  # (citing arxiv_id, cited base id)
        self.contexts: list[dict] = []
        self.ranks: dict[str, dict] = {}

    def add_paper(
        self,
        arxiv_id: str,
        title: str = "",
        published: str = "2024-01-01",
        category: str = "cs.LG",
    ) -> None:
        self.papers[arxiv_id] = {
            "title": title or f"Title {arxiv_id}",
            "published": published,
            "category": category,
        }

    def cite(self, citing: str, cited: str, function: str = "neutral", sentence: str = "") -> None:
        self.edges.add((citing, base_id(cited)))
        if sentence:
            self.contexts.append(
                {
                    "citing_id": citing,
                    "cited_base_id": base_id(cited),
                    "sentence": sentence,
                    "function": function,
                    "anchor": None,
                }
            )

    def _resolve(self, base: str) -> str | None:
        return next((a for a in self.papers if base_id(a) == base), None)

    async def corpus_graph(self):
        edges = [(c, d) for c, b in self.edges if (d := self._resolve(b)) and c in self.papers]
        return list(self.papers), edges

    async def save_influence(self, rows):
        self.ranks = {r["arxiv_id"]: r for r in rows}

    async def influence(self, arxiv_ids):
        return {i: self.ranks[i]["pagerank"] for i in arxiv_ids if i in self.ranks}

    async def paper(self, arxiv_id):
        pid = arxiv_id if arxiv_id in self.papers else self._resolve(arxiv_id)
        if pid is None:
            return None
        rank = self.ranks.get(pid, {})
        return {
            "arxiv_id": pid,
            **self.papers[pid],
            "in_degree": rank.get("in_degree", 0),
            "pagerank": rank.get("pagerank", 0.0),
            "reference_count": sum(1 for c, _ in self.edges if c == pid),
            "cited_by": sum(1 for _, b in self.edges if b == base_id(pid)),
        }

    async def prerequisites(self, arxiv_id, max_depth, limit):
        found: dict[str, dict] = {}
        frontier = [(arxiv_id, [arxiv_id])]
        for depth in range(1, max_depth + 1):
            nxt = []
            for node, path in frontier:
                for citing, base in sorted(self.edges):
                    target = self._resolve(base)
                    if citing == node and target and target not in path and target not in found:
                        found[target] = {
                            "arxiv_id": target,
                            "title": self.papers[target]["title"],
                            "depth": depth,
                            "path": [*path, target],
                            "pagerank": self.ranks.get(target, {}).get("pagerank", 0.0),
                            "published": self.papers[target]["published"],
                        }
                        nxt.append((target, [*path, target]))
            frontier = nxt
        ranked = sorted(found.values(), key=lambda r: (r["depth"], -r["pagerank"], r["published"]))
        return ranked[:limit]

    async def citing_contexts(self, arxiv_id, functions, limit):
        rows = [
            {**c, "citing_title": self.papers.get(c["citing_id"], {}).get("title")}
            for c in self.contexts
            if c["cited_base_id"] == base_id(arxiv_id)
            and (functions is None or c["function"] in functions)
        ]
        rows.sort(key=lambda c: FUNCTION_PRIORITY.index(c["function"]))
        return rows[:limit]

    async def neighbors(self, arxiv_ids, limit):
        bases = {base_id(i) for i in arxiv_ids}
        out = {self._resolve(b) for c, b in self.edges if c in arxiv_ids}
        inc = {c for c, b in self.edges if b in bases}
        return sorted(n for n in (out | inc) if n and n not in arxiv_ids)[:limit]

    async def neighborhood(self, arxiv_id, depth, max_nodes):
        if arxiv_id not in self.papers:
            return {"nodes": [], "edges": [], "truncated": False}
        ids = {arxiv_id, *await self.neighbors([arxiv_id], max_nodes)}
        nodes = [{"id": i, "title": self.papers[i]["title"], "in_corpus": True} for i in ids]
        edges = [
            {"source": c, "target": self._resolve(b), "function": "neutral"}
            for c, b in self.edges
            if c in ids and self._resolve(b) in ids
        ]
        return {"nodes": nodes, "edges": edges, "truncated": False}

    async def top_influential(self, limit):
        rows = sorted(self.ranks.values(), key=lambda r: -r["pagerank"])[:limit]
        return [{**r, "title": self.papers[r["arxiv_id"]]["title"]} for r in rows]

    async def map(self, limit):
        ranked = sorted(
            self.papers,
            key=lambda p: -self.ranks.get(p, {}).get("pagerank", 0.0),
        )[:limit]
        nodes = [
            {
                "id": p,
                "title": self.papers[p]["title"],
                "published": self.papers[p]["published"],
                "category": self.papers[p].get("category"),
                "pagerank": self.ranks.get(p, {}).get("pagerank", 0.0),
                "in_degree": self.ranks.get(p, {}).get("in_degree", 0),
            }
            for p in ranked
        ]
        functions: dict[tuple[str, str], list[str]] = {}
        for c in self.contexts:
            functions.setdefault((c["citing_id"], c["cited_base_id"]), []).append(c["function"])
        edges = [
            {"source": c, "target": t, "function": _strongest(functions.get((c, b)))}
            for c, b in sorted(self.edges)
            if c in ranked and (t := self._resolve(b)) in ranked and t != c
        ]
        return {"nodes": nodes, "edges": edges}

    async def missing(self, limit):
        counts: dict[str, int] = {}
        for _, base in self.edges:
            if self._resolve(base) is None:
                counts[base] = counts.get(base, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [{"arxiv_id": b, "cited_by": n} for b, n in ranked]
