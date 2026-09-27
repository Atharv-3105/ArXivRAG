"""Turning search hits into numbered sources (shared by retrieval and the specialists)."""

from src.agents.state import Source
from src.core.config import settings
from src.storage.vector_db import SearchHit


def retrieval_query(query: str, history: list[dict]) -> str:
    """Short follow-ups ("and its limitations?") are searched together with the previous user
    message. An LLM rewrite would be better but costs a call per turn on a free quota."""
    if len(query.split()) > settings.FOLLOWUP_MAX_WORDS:
        return query
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    return f"{previous}\n{query}" if previous else query


def url_for(hit: SearchHit) -> str:
    anchor = hit.metadata.get("anchor")
    if anchor and hit.metadata.get("parse_method") == "html":
        return f"https://arxiv.org/html/{hit.paper_id}#{anchor}"
    return f"https://arxiv.org/abs/{hit.paper_id}"


def to_source(n: int, hit: SearchHit, role: str = "retrieved") -> Source:
    return Source(
        n=n,
        chunk_id=hit.chunk_id,
        paper_id=hit.paper_id,
        title=hit.title,
        section=hit.section,
        content=hit.content,
        score=round(hit.score, 4),
        url=url_for(hit),
        math_fidelity=hit.metadata.get("math_fidelity", "degraded"),
        role=role,
        published=hit.published,
    )


def select_sources(hits: list[SearchHit], max_per_paper: int | None = None) -> list[Source]:
    """Keep ranking order, cap chunks per paper so one long paper cannot fill the context."""
    cap = max_per_paper or settings.RETRIEVAL_MAX_PER_PAPER
    per_paper: dict[str, int] = {}
    sources: list[Source] = []
    for hit in hits:
        if per_paper.get(hit.paper_id, 0) >= cap:
            continue
        per_paper[hit.paper_id] = per_paper.get(hit.paper_id, 0) + 1
        sources.append(to_source(len(sources) + 1, hit))
    return sources
