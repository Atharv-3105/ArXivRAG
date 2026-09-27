"""Prompt templates: versioned Jinja2 files, A/B variants, and the shared source/reader blocks.

Templates live in src/agents/templates as `<name>.v<N>.j2` (e.g. tutor_explain.v2.j2). Every
answer records `<name>@v<N>` (messages.prompt_version), so a change in answer quality can be traced
to the template that produced it and variants can be compared (Phase 6 eval).

A/B: PROMPT_VARIANTS lists the versions in rotation per template. A user always gets the same
variant (sha256 of user id + template name), so the experience is consistent and the comparison
is between users, not between one user's turns.

Validation: StrictUndefined makes a missing variable an error instead of an empty string, and
`validate_templates()` runs at startup, so a broken template or a misconfigured variant stops the
app from starting rather than failing on some user's turn.
"""

import hashlib
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from src.agents.state import Source
from src.core.config import settings
from src.llm.base import ChatMessage

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_FILE = re.compile(r"^(?P<name>[a-z_]+)\.(?P<version>v\d+)\.j2$")

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    autoescape=False,  # plain-text prompts, not HTML
)


def available() -> dict[str, list[str]]:
    """template name -> versions on disk, oldest first."""
    found: dict[str, list[str]] = {}
    for path in TEMPLATE_DIR.glob("*.j2"):
        m = _FILE.match(path.name)
        if m:
            found.setdefault(m["name"], []).append(m["version"])
    return {k: sorted(v, key=lambda s: int(s[1:])) for k, v in found.items()}


def choose_version(name: str, user_id: int | None) -> str:
    variants = settings.PROMPT_VARIANTS.get(name) or [available()[name][-1]]
    if not settings.PROMPT_AB_TEST or user_id is None or len(variants) == 1:
        return variants[0]
    digest = hashlib.sha256(f"{user_id}:{name}".encode()).digest()
    return variants[int.from_bytes(digest[:4], "big") % len(variants)]


def render(name: str, user_id: int | None = None, **variables) -> tuple[str, str]:
    """(prompt text, "name@version")."""
    version = choose_version(name, user_id)
    text = _env.get_template(f"{name}.{version}.j2").render(**variables)
    return re.sub(r"\n{3,}", "\n\n", text).strip(), f"{name}@{version}"


def validate_templates() -> dict[str, list[str]]:
    """Compile every template and check every configured variant exists. Raises on problems."""
    versions = available()
    for name, vs in versions.items():
        for v in vs:
            _env.get_template(f"{name}.{v}.j2")
    for name, variants in settings.PROMPT_VARIANTS.items():
        for v in variants:
            if v not in versions.get(name, []):
                raise TemplateNotFound(f"PROMPT_VARIANTS names {name}.{v}.j2, which does not exist")
    return versions


# ------------------------------------------------------------------ shared blocks

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
_ROLE_LABEL = {
    "prerequisite": " (prerequisite)",
    "related": " (related via citations)",
    "citation_context": " (what another paper says)",
}


def reader_lines(profile: dict | None, mastery: dict[str, float] | None) -> list[str]:
    """How to pitch the answer. Empty when we know nothing about the reader."""
    if not profile:
        return []
    level = _LEVEL.get(profile.get("expertise_level") or "", _LEVEL["intermediate"])
    lines = [
        f"The reader is {level}.",
        _MATH.get(profile.get("math_depth") or "", _MATH["standard"]),
    ]
    if profile.get("goals"):
        lines.append("Their goals: " + "; ".join(profile["goals"][:3]) + ".")
    mastery = mastery or {}
    strong = [c.replace("_", " ") for c, m in mastery.items() if m >= settings.MASTERY_STRONG_ABOVE]
    weak = [c.replace("_", " ") for c, m in mastery.items() if m < settings.MASTERY_WEAK_BELOW]
    if strong:
        lines.append("They already know: " + ", ".join(strong) + " (do not re-explain).")
    if weak:
        lines.append("Still shaky on: " + ", ".join(weak) + " (explain these carefully).")
    return lines


def format_sources(sources: list[Source]) -> str:
    """One block per source. The mock LLM parses lines that start with "[n]"."""
    if not sources:
        return "(no sources were retrieved)"
    blocks = []
    for s in sources:
        tags = " (math: degraded)" if s["math_fidelity"] == "degraded" else ""
        tags += _ROLE_LABEL.get(s.get("role", "retrieved"), "")
        year = f", {(s.get('published') or '')[:4]}" if s.get("published") else ""
        body = " ".join(s["content"].split())
        blocks.append(
            f"[{s['n']}] {body}\n    — {s['title']} ({s['paper_id']}{year}), {s['section']}{tags}"
        )
    return "\n\n".join(blocks)


def build_messages(
    system: str, sources: list[Source], history: list[dict], user_message: str
) -> list[ChatMessage]:
    messages = [ChatMessage("system", f"{system}\n\nSources:\n{format_sources(sources)}")]
    for m in history:
        if m["role"] in ("user", "assistant") and m["content"]:
            messages.append(ChatMessage(m["role"], m["content"]))
    messages.append(ChatMessage("user", user_message))
    return messages
