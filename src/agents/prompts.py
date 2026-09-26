"""Prompt assembly for the generate node.

Phase 2 keeps one grounded prompt with a per-intent instruction. Versioned Jinja templates, the
Socratic tutor and A/B variants arrive in Phase 5; PROMPT_VERSION is logged with every answer so
quality can be compared across versions from day one.
"""

from src.agents.state import Intent, Source
from src.llm.base import ChatMessage

PROMPT_VERSION = "chat-v2"  # v2: reader profile block (Phase 3)

_BASE = """You are ArXivMind, a research assistant for AI/ML papers.

Rules:
- Answer ONLY from the numbered sources below. Cite every claim with its source number, e.g. [2].
- If the sources do not contain the answer, say so plainly. Never invent papers, results or numbers.
- Keep LaTeX as LaTeX ($...$ inline, $$...$$ display).
- Sources marked (math: degraded) were extracted from a PDF: their formulas may be garbled, so do
  not quote those formulas as exact.
- Be concise. Use short paragraphs or bullet points."""

_INTENT: dict[Intent, str] = {
    "learn": (
        "The user wants to understand a concept. Build intuition first, then the precise idea, "
        "then (if relevant) the formula with each symbol explained."
    ),
    "critique": (
        "The user wants a critical assessment. Cover assumptions, baselines, statistical "
        "validity and reproducibility. Ground every criticism in the cited text; label anything "
        "the sources do not settle as an open question rather than a flaw."
    ),
    "synthesize": (
        "The user wants several works compared. Say where the sources agree, where they "
        "contradict each other (cite both sides), and what remains open. A compact table is "
        "welcome when it helps. Do not treat different setups as directly comparable."
    ),
    "recommend": (
        "The user wants to know what to read. Recommend papers from the sources only, each "
        "with one line on why it is the right next step."
    ),
    "chat": "Answer helpfully and briefly.",
}


def format_sources(sources: list[Source]) -> str:
    """One block per source. The mock LLM parses lines that start with "[n]"."""
    if not sources:
        return "(no sources were retrieved)"
    blocks = []
    for s in sources:
        fidelity = " (math: degraded)" if s["math_fidelity"] == "degraded" else ""
        body = " ".join(s["content"].split())
        blocks.append(
            f"[{s['n']}] {body}\n    — {s['title']} ({s['paper_id']}), {s['section']}{fidelity}"
        )
    return "\n\n".join(blocks)


_LEVEL = {
    "beginner": "a beginner: define every term, prefer analogies, introduce notation gently",
    "intermediate": "comfortable with ML basics: skip fundamentals, explain the key idea precisely",
    "advanced": "an expert: be dense and technical, focus on subtleties and trade-offs",
}
_MATH = {
    "light": "Keep formulas to a minimum; explain in words.",
    "standard": "Use formulas where they help, and explain each symbol.",
    "rigorous": "Use full mathematical detail and precise statements.",
}


def reader_block(profile: dict, mastery: dict[str, float]) -> str:
    """How to pitch the answer. Empty when we know nothing about the reader."""
    if not profile:
        return ""
    level = _LEVEL.get(profile.get("expertise_level", ""), _LEVEL["intermediate"])
    lines = [f"The reader is {level}.", _MATH.get(profile.get("math_depth", ""), _MATH["standard"])]
    if profile.get("goals"):
        lines.append("Their goals: " + "; ".join(profile["goals"][:3]) + ".")
    strong = [c.replace("_", " ") for c, m in mastery.items() if m >= 0.7]
    weak = [c.replace("_", " ") for c, m in mastery.items() if m < 0.35]
    if strong:
        lines.append("They already know: " + ", ".join(strong) + " (do not re-explain).")
    if weak:
        lines.append("Still shaky on: " + ", ".join(weak) + " (explain these carefully).")
    return "\n\nReader:\n" + "\n".join(f"- {line}" for line in lines)


def build_messages(
    query: str,
    intent: Intent,
    sources: list[Source],
    history: list[dict],
    profile: dict | None = None,
    mastery: dict[str, float] | None = None,
) -> list[ChatMessage]:
    system = (
        f"{_BASE}\n\nTask: {_INTENT[intent]}{reader_block(profile or {}, mastery or {})}"
        f"\n\nSources:\n{format_sources(sources)}"
    )
    messages = [ChatMessage("system", system)]
    for m in history:
        if m["role"] in ("user", "assistant") and m["content"]:
            messages.append(ChatMessage(m["role"], m["content"]))
    messages.append(ChatMessage("user", query))
    return messages
