"""Extract arXiv identifiers from a paper's reference list -> citation edges.

The arXiv API gives no references, so edges come from the parsed bibliography (HTML tier). A
reference that names an arXiv paper (``arXiv:2401.12345``, ``arxiv.org/abs/2401.12345v2``,
``hep-th/9901001``) becomes an edge to that paper's *base* id (no version), because citations almost
never name a version.
"""

import re

from src.pipeline.models import Reference

# New-style ids (since 2007): YYMM.NNNN or YYMM.NNNNN, optional version.
# Not preceded by a digit or "digit." (so "12.2401.12345" page ranges do not match).
_NEW = r"(?<!\d)(?<!\d\.)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)"
# Old-style ids: archive(.subject)/YYMMNNN
_OLD = r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b"

_ARXIV_CONTEXT = re.compile(rf"(?:arxiv[\s:./]*(?:abs/|pdf/)?\s*)(?:{_NEW}|{_OLD})", re.IGNORECASE)
# PDF/HTML extraction sometimes breaks URLs with a space ("arxiv. org/abs/...").
_URL = re.compile(rf"arxiv\.\s?org/(?:abs|pdf|html)/(?:{_NEW}|{_OLD})", re.IGNORECASE)
# arXiv's own DOI prefix: 10.48550/arXiv.2211.03154
_DOI = re.compile(rf"10\.48550/arxiv\.(?:{_NEW}|{_OLD})", re.IGNORECASE)

# Semantic Scholar style "ArXiv, vol. abs/2009.04372" - only trusted when the reference says arXiv.
_ABS = re.compile(rf"\babs/(?:{_NEW}|{_OLD})", re.IGNORECASE)


def base_id(arxiv_id: str) -> str:
    """'2401.12345v2' -> '2401.12345'."""
    return re.sub(r"v\d+$", "", arxiv_id)


def extract_arxiv_ids(text: str, urls: list[str] | None = None) -> list[str]:
    """arXiv base ids mentioned in one reference, in order of first appearance, deduplicated.

    Only ids next to an "arXiv" marker or inside an arxiv.org URL count: a bare "2401.12345" in a
    reference is as likely to be a page range or DOI fragment as an arXiv id.
    """
    found: list[str] = []
    sources = [text, *(urls or [])]
    for source in sources:
        patterns = [_ARXIV_CONTEXT, _URL, _DOI]
        if "arxiv" in source.lower():
            patterns.append(_ABS)
        for pattern in patterns:
            for m in pattern.finditer(source):
                ident = m.group(1) or m.group(2)
                if ident and ident not in found:
                    found.append(ident)
    return found


def citation_edges(citing_id: str, references: list[Reference]) -> list[dict]:
    """Rows for paper_citations. A paper citing another version of itself is dropped."""
    own = base_id(citing_id)
    edges: dict[str, dict] = {}
    for ref in references:
        for cited in extract_arxiv_ids(ref.text, ref.urls):
            if cited == own or cited in edges:
                continue
            edges[cited] = {
                "citing_id": citing_id,
                "cited_base_id": cited,
                "ref_ordinal": ref.ordinal,
            }
    return list(edges.values())
