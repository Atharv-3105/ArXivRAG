"""Extract arXiv identifiers from a paper's reference list -> citation edges.

The arXiv API gives no references, so edges come from the parsed bibliography (HTML tier). A
reference that names an arXiv paper (``arXiv:2401.12345``, ``arxiv.org/abs/2401.12345v2``,
``hep-th/9901001``) becomes an edge to that paper's *base* id (no version), because citations almost
never name a version.
"""

import re
from collections import defaultdict

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


# ---------------------------------------------------------------- citation function (ADR 0009)
#
# What a citing sentence *does* with the cited work, after Teufel et al.'s citation-function
# scheme, collapsed to four classes the product needs:
#   critical     points out a weakness / failure / disagreement      -> "who criticises X"
#   comparative  positions results against it ("outperforms [12]")   -> "how X compares"
#   builds_on    uses or extends it ("following [12]", "we adopt")   -> prerequisite chains
#   neutral      everything else (background mentions)
# Rules, not a model: free, deterministic, explainable. Precision over recall: "however" alone
# is too common, so contrast words only count next to a weakness word. Measured in Phase 6.

CITATION_FUNCTIONS = ("critical", "comparative", "builds_on", "neutral")

_CRITICAL = re.compile(
    r"\b(fail(s|ed|ure)?( to)?|suffer(s|ed)? from|limitation|drawback|shortcoming|flaw(ed)?|"
    r"problematic|incorrect|erroneous|unrealistic|does not (hold|scale|generali[sz]e)|"
    r"cannot|can ?not|unable to|not robust|brittle|overestimate|underestimate|"
    r"contrary to|contradict|questionable|we (show|find|argue) that .{0,60}\b(not|no)\b|"
    r"is (limited|restricted) to|overlook(s|ed)?)\b",
    re.IGNORECASE,
)
_COMPARATIVE = re.compile(
    r"\b(outperform(s|ed|ing)?|surpass(es|ed)?|improv(e|es|ed|ing) (up)?on|better than|"
    r"worse than|compared (to|with)|in contrast to|unlike|than (the )?(prior|previous|baseline)|"
    r"state[- ]of[- ]the[- ]art|beats?|competitive with|on par with)\b",
    re.IGNORECASE,
)
_BUILDS_ON = re.compile(
    r"\b(follow(ing|s)?|build(s|ing)? (up)?on|built on|based on|extend(s|ed|ing)?|inspired by|"
    r"we (use|adopt|employ|leverage|apply|borrow|reuse)|adopt(ed|ing)? from|as (in|proposed in)|"
    r"introduced (in|by)|proposed (in|by)|originally (proposed|introduced))\b",
    re.IGNORECASE,
)


def citation_function(sentence: str) -> str:
    if _CRITICAL.search(sentence):
        return "critical"
    if _COMPARATIVE.search(sentence):
        return "comparative"
    if _BUILDS_ON.search(sentence):
        return "builds_on"
    return "neutral"


def context_rows(citing_id: str, references: list[Reference], contexts: list) -> list[dict]:
    """Rows for citation_contexts: one per (sentence, cited arXiv paper). Contexts whose keys
    resolve to no arXiv id are dropped: they cannot join the graph."""
    own = base_id(citing_id)
    by_key = {r.key: r for r in references if r.key}
    cited_in: dict[str, set[str]] = defaultdict(set)  # sentence -> bibliography keys it cites
    for ctx in contexts:
        cited_in[ctx.sentence].update(ctx.keys)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for ctx in contexts:
        # A stance can only be attributed to a work the sentence is about. A sentence that cites a
        # group ("... (A, 2020; B, 2021; C, 2022)") is background, and keyword rules mislabelled
        # it constantly: on real text only ~1 in 4 non-neutral labels was right, and the wrong ones
        # were nearly all multi-citation sentences. The parser emits one context per citation
        # anchor, so the group is found by counting the distinct keys per sentence.
        function = (
            citation_function(ctx.sentence) if len(cited_in[ctx.sentence]) == 1 else "neutral"
        )
        for key in ctx.keys:
            ref = by_key.get(key)
            if ref is None:
                continue
            for cited in extract_arxiv_ids(ref.text, ref.urls):
                if cited == own or (ctx.sentence, cited) in seen:
                    continue
                seen.add((ctx.sentence, cited))
                rows.append(
                    {
                        "citing_id": citing_id,
                        "cited_base_id": cited,
                        "ref_ordinal": ref.ordinal,
                        "sentence": ctx.sentence,
                        "anchor": ctx.anchor,
                        "function": function,
                    }
                )
    return rows
