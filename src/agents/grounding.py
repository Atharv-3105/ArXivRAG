"""Map [n] markers in an answer back to retrieved sources.

This is the cheap, deterministic half of grounding: it catches citations to sources that were
never retrieved (a classic hallucination). Checking that a cited source actually *supports* the
claim needs an LLM or NLI judge - that is the Guardrail's job (Phase 6).
"""

import re

from src.agents.state import Citation, Source

# [3], [1, 2], [1][2], [2-4]. Not "a[0]" (indexing in code/math): no word character before "[".
_MARKER = re.compile(r"(?<!\w)\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")


def cited_numbers(text: str) -> list[int]:
    found: list[int] = []
    for group in _MARKER.findall(text):
        for part in re.split(r"\s*,\s*", group):
            bounds = re.split(r"\s*[\-–]\s*", part)
            if len(bounds) == 2:
                lo, hi = int(bounds[0]), int(bounds[1])
                numbers = range(lo, hi + 1) if 0 < hi - lo < 20 else [lo, hi]
            else:
                numbers = [int(bounds[0])]
            for n in numbers:
                if n > 0 and n not in found:
                    found.append(n)
    return found


def resolve_citations(text: str, sources: list[Source]) -> tuple[list[Citation], list[int]]:
    """(citations in order of first use, numbers that match no source)."""
    by_n = {s["n"]: s for s in sources}
    citations: list[Citation] = []
    ungrounded: list[int] = []
    for n in cited_numbers(text):
        source = by_n.get(n)
        if source is None:
            ungrounded.append(n)
            continue
        citations.append(
            Citation(
                n=n,
                paper_id=source["paper_id"],
                title=source["title"],
                section=source["section"],
                url=source["url"],
            )
        )
    return citations, ungrounded
