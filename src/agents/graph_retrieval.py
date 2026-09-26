"""Graph-enhanced retrieval: fuse the citation graph into what the vector search found (ADR 0009).

Two steps, both optional and both bounded by GRAPH_TIMEOUT_SECONDS (the caller falls back to the
plain vector result on any failure):

1. Re-rank by the reader. RRF gives a rank; we turn it into 1/(10 + rank) and multiply boosts:
     weak reader   (low mastery / beginner):  background-type sections, influential papers
     strong reader (high mastery / advanced): method/experiment sections, recent papers
   Boosts are capped (1.5x) and the rank curve is steep enough that they reorder neighbours,
   never promote a low-ranked chunk over the clear winner.

2. Expand by intent, using edges the vector index cannot see:
     learn + weak reader -> prerequisites: in-corpus papers the top result builds on
     critique            -> what other papers say about the top results (critical/comparative)
     synthesize          -> papers connected to the top results by citations
"""

import re
from datetime import UTC, date, datetime

from src.agents.state import Source
from src.core.config import settings
from src.storage.vector_db import SearchHit

_BACKGROUND = re.compile(
    r"\b(abstract|introduction|background|preliminar\w*|related work|overview|motivation)\b", re.I
)
_ADVANCED = re.compile(
    r"\b(method\w*|approach|architecture|experiment\w*|results?|analysis|ablation\w*|"
    r"implementation|evaluation|theor\w*|proof\w*)\b",
    re.I,
)
RECENT_YEARS = 2
RECENCY_BOOST = 1.1
# Base score 1/(RANK_K + rank). A boost b lets a hit overtake at most about (b - 1)(RANK_K + rank)
# positions. RRF's usual k=60 is far too flat for this: a 1.25x boost would jump ~15 places and
# the "gentle" re-rank would reorder everything (a test caught exactly that). With k=10 and the
# combined boost capped at 1.5, the top hit can only be passed by the next ~5.
RANK_K = 10
MAX_BOOST = 1.5


def mastery_level(profile: dict | None, mastery: dict[str, float] | None) -> str | None:
    """'weak' | 'strong' | None, from mastery of the query's concepts, else the profile level."""
    values = list((mastery or {}).values())
    if values:
        mean = sum(values) / len(values)
        if mean < settings.MASTERY_WEAK_BELOW:
            return "weak"
        if mean > settings.MASTERY_STRONG_ABOVE:
            return "strong"
        return None
    level = (profile or {}).get("expertise_level")
    return {"beginner": "weak", "advanced": "strong"}.get(level or "")


def _recent(published: str | None, today: date) -> bool:
    if not published:
        return False
    try:
        return (today - date.fromisoformat(published[:10])).days <= RECENT_YEARS * 365
    except ValueError:
        return False


def rerank(
    hits: list[SearchHit],
    level: str | None,
    influence: dict[str, float],
    today: date | None = None,
) -> list[SearchHit]:
    today = today or datetime.now(UTC).date()
    boost_section = settings.RERANK_SECTION_BOOST
    weight = settings.RERANK_INFLUENCE_WEIGHT

    def score(item: tuple[int, SearchHit]) -> float:
        rank, hit = item
        rank_influence = min(influence.get(hit.paper_id, 0.0), 5.0)
        boost = 1.0
        if level == "weak":
            if _BACKGROUND.search(hit.section):
                boost *= boost_section
            boost *= 1 + weight * rank_influence
        elif level == "strong":
            if _ADVANCED.search(hit.section):
                boost *= boost_section
            if _recent(hit.published, today):
                boost *= RECENCY_BOOST
        else:
            boost *= 1 + weight / 2 * rank_influence
        return min(boost, MAX_BOOST) / (RANK_K + rank)

    return [hit for _, hit in sorted(enumerate(hits), key=score, reverse=True)]


def _source(n: int, hit: SearchHit, role: str, url: str) -> Source:
    return Source(
        n=n,
        chunk_id=hit.chunk_id,
        paper_id=hit.paper_id,
        title=hit.title,
        section=hit.section,
        content=hit.content,
        score=round(hit.score, 4),
        url=url,
        math_fidelity=hit.metadata.get("math_fidelity", "degraded"),
        role=role,
    )


async def expand(
    intent: str,
    query: str,
    sources: list[Source],
    level: str | None,
    graph,
    retriever,
    url_for,
) -> list[Source]:
    """Extra sources from the citation graph, numbered after the existing ones."""
    if not sources:
        return []
    budget = settings.GRAPH_CONTEXT_PAPERS
    have = {s["paper_id"] for s in sources}
    top = list(dict.fromkeys(s["paper_id"] for s in sources))
    extra: list[Source] = []

    async def best_chunks(paper_ids: list[str], role: str) -> None:
        if not paper_ids:
            return
        hits = await retriever.search(query, top_k=len(paper_ids) * 3, paper_ids=paper_ids)
        seen: set[str] = set()
        for hit in hits:
            if hit.paper_id in seen or len(seen) >= budget:
                continue
            seen.add(hit.paper_id)
            extra.append(_source(len(sources) + len(extra) + 1, hit, role, url_for(hit)))

    if intent == "learn" and level == "weak":
        chain = await graph.prerequisites(top[0], settings.GRAPH_MAX_DEPTH, budget * 2)
        await best_chunks(
            [p["arxiv_id"] for p in chain if p["arxiv_id"] not in have][:budget], "prerequisite"
        )
    elif intent == "critique":
        for paper_id in top[:2]:
            title = next(s["title"] for s in sources if s["paper_id"] == paper_id)
            for ctx in await graph.citing_contexts(paper_id, ["critical", "comparative"], budget):
                if len(extra) >= budget:
                    break
                extra.append(
                    Source(
                        n=len(sources) + len(extra) + 1,
                        chunk_id=f"ctx:{ctx['citing_id']}:{paper_id}:{len(extra)}",
                        paper_id=ctx["citing_id"],
                        title=ctx.get("citing_title") or ctx["citing_id"],
                        section=f"on {title}",
                        content=ctx["sentence"],
                        score=0.0,
                        url=f"https://arxiv.org/abs/{ctx['citing_id']}",
                        math_fidelity="latex",
                        role="citation_context",
                    )
                )
    elif intent == "synthesize":
        related = await graph.neighbors(top[:3], budget * 2)
        await best_chunks([p for p in related if p not in have][:budget], "related")
    return extra
