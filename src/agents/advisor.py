"""Advisor: what to read next. Ranking is deterministic; no LLM call (ADR 0010).

Why no LLM: the ranking signals are all data we already have (retrieval rank, the reader's weak
concepts, citation-graph prerequisites, PageRank, interests), and an LLM asked to "rank these"
would be slower, costlier and unexplainable. Every recommendation carries the reasons that put it
there, generated from the same numbers that ranked it.

    score = base(rank) x (1 + 0.5·weak concepts covered) x (1 + 0.3·papers it is a prerequisite of)
                       x (1 + 0.1·min(PageRank, 5)) x (1 + 0.2·interests matched)
"""

from dataclasses import dataclass, field

from src.memory import concepts as vocab

RANK_K = 5  # base = 1 / (RANK_K + rank): retrieval order matters, but signals can reorder


@dataclass
class Candidate:
    paper_id: str
    title: str
    snippet: str
    rank: int  # retrieval rank; prerequisites that were not retrieved get len(candidates)
    published: str | None = None
    url: str = ""
    concepts: list[str] = field(default_factory=list)
    prerequisite_of: list[str] = field(default_factory=list)  # titles of papers that build on it
    pagerank: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def rank_candidates(
    candidates: list[Candidate],
    weak: set[str],
    interests: set[str],
    limit: int,
) -> list[Candidate]:
    for c in candidates:
        covered_weak = [x for x in c.concepts if x in weak]
        matched = [x for x in c.concepts if x in interests]
        c.score = (
            (1.0 / (RANK_K + c.rank))
            * (1 + 0.5 * len(covered_weak))
            * (1 + 0.3 * len(c.prerequisite_of))
            * (1 + 0.1 * min(c.pagerank, 5.0))
            * (1 + 0.2 * len(matched))
        )
        if c.prerequisite_of:
            c.reasons.append("foundational: " + ", ".join(c.prerequisite_of[:2]) + " build on it")
        if covered_weak:
            c.reasons.append(
                "covers "
                + ", ".join(_label(x) for x in covered_weak)
                + ", which you are still building"
            )
        if c.pagerank >= 1.5:
            c.reasons.append(f"one of the most influential papers here ({c.pagerank:.1f}x average)")
        if matched:
            c.reasons.append("matches your interest in " + ", ".join(_label(x) for x in matched))
        if not c.reasons:
            c.reasons.append("closely matches what you asked about")
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    return ranked[:limit]


def _label(concept: str) -> str:
    return vocab.label(concept)


def render(recommendations: list[Candidate], due: list[str], weak: list[str]) -> str:
    """The Advisor's reply. Each recommendation is cited [n] so grounding works as elsewhere."""
    if not recommendations:
        lines = ["I couldn't find papers to recommend for this in the indexed corpus yet."]
    else:
        lines = ["Here is what I'd read next, in order:", ""]
        for n, c in enumerate(recommendations, start=1):
            year = f", {c.published[:4]}" if c.published else ""
            lines.append(f"{n}. **{c.title}** ({c.paper_id}{year}): {'; '.join(c.reasons)} [{n}]")
    if weak:
        lines += ["", "Concepts to strengthen: " + ", ".join(_label(w) for w in weak[:5]) + "."]
    if due:
        lines += [
            "",
            f"You also have {len(due)} concept(s) due for review: "
            + ", ".join(_label(d) for d in due[:5])
            + '. Say "quiz me" to start.',
        ]
    return "\n".join(lines)
