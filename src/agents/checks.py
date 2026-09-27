"""Deterministic output checks per specialist (ADR 0010).

Checks run on every answer and are recorded with it (messages.checks, the API response, logs).
They are not retried: the text has usually already been streamed to the reader, and regenerating
would show them an answer and then replace it. Instead each failure is either *repaired*
deterministically (e.g. several questions -> the last one becomes the pending question) or
*measured* - the Phase 6 eval tracks violation rates per template version and model.
"""

import re

_MATH = re.compile(r"\$\$.+?\$\$|\$[^$]+\$", re.DOTALL)
_CODE = re.compile(r"```.+?```|`[^`]+`", re.DOTALL)
_CITATION = re.compile(r"(?<!\w)\[\d+(?:\s*[,\-–]\s*\d+)*\]")

CRITIC_SECTIONS = (
    "Summary",
    "Strengths",
    "Weaknesses",
    "Assumptions",
    "Baselines and evaluation",
    "Reproducibility",
    "Open questions",
)
SYNTH_SECTIONS = (
    "Comparison",
    "Where they agree",
    "Where they disagree",
    "Timeline",
    "Open directions",
)


def _prose(text: str) -> str:
    return _CODE.sub(" ", _MATH.sub(" ", text))


def questions(text: str) -> list[str]:
    """Sentences ending in '?' outside math and code."""
    return [q.strip() for q in re.findall(r"[^.!?\n]*\?", _prose(text)) if len(q.strip()) > 3]


def last_question(text: str) -> str | None:
    found = questions(text)
    return found[-1] if found else None


def _sections(text: str) -> dict[str, str]:
    """Markdown '## Heading' -> body."""
    parts = re.split(r"^#{2,3}\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return {parts[i].strip().lower(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _bullets(body: str) -> list[str]:
    return [b for b in re.findall(r"^\s*(?:[-*]|\d+\.)\s+(.+)$", body, re.MULTILINE)]


def tutor_checks(text: str, meta: dict | None, expects_question: bool) -> dict[str, bool]:
    n = len(questions(text))
    result = {"trailer_parsed": meta is not None}
    if expects_question:
        result["one_question"] = n == 1
    else:
        result["no_question"] = n == 0
    return result


def critic_checks(text: str) -> dict[str, bool]:
    sections = _sections(text)
    weaknesses = _bullets(sections.get("weaknesses", ""))
    strengths = _bullets(sections.get("strengths", ""))
    return {
        "all_sections": all(s.lower() in sections for s in CRITIC_SECTIONS),
        "claims_cited": bool(weaknesses + strengths)
        and all(_CITATION.search(b) for b in weaknesses + strengths),
    }


def synthesizer_checks(text: str, cited_papers: set[str]) -> dict[str, bool]:
    sections = _sections(text)
    return {
        "all_sections": all(s.lower() in sections for s in SYNTH_SECTIONS),
        "comparison_table": bool(re.search(r"^\s*\|?\s*:?-{3,}", text, re.MULTILINE)),
        "two_or_more_papers": len(cited_papers) >= 2,
    }
