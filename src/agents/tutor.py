"""The Tutor's session logic: pure functions, no I/O (ADR 0010).

A tutoring session is multi-turn, so it has state, saved with the conversation
(conversations.state):

    mode        explain | review | reading | equation
    topic       what the session is about
    concept     vocabulary slug, if any (mastery updates need one)
    step        check questions asked so far (session ends at TUTOR_MAX_STEPS)
    level       0 normal, -1 simpler, -2 simplest (each "I don't understand" steps down)
    hints_used  hints given for the pending question
    pending     {"question", "answer_key"} or None
    sources     compact copies of the sources the question was built on (for grading)
    reading     {"paper_id", "title", "index", "total", "section"} in reading mode
    equation    {"paper_id", "number", "latex", "section"} in equation mode

Actions:
    explain    a new learning question                    (router: intent learn)
    grade      the reader answered the pending question   (router: pending + not a new question)
    hint       "hint"                                     (needs a pending question)
    reexplain  "I don't understand"                       (steps the level down)
    skip       "skip" / "I give up"                        (no LLM call)
    review     "quiz me" / due spaced-repetition reviews
    read       "walk me through <arXiv id>"               (guided reading, section by section)
    read_next  "next" while reading
    equation   "explain equation 3 of <arXiv id>"
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.agents.checks import last_question
from src.core.config import settings
from src.memory import concepts as vocab

ACTIONS = (
    "explain",
    "grade",
    "hint",
    "reexplain",
    "skip",
    "review",
    "read",
    "read_next",
    "equation",
)
# What the API's explicit `action` field maps to (UI buttons send these; no guessing needed).
EXPLICIT = {
    "answer": "grade",
    "hint": "hint",
    "dont_understand": "reexplain",
    "skip": "skip",
    "review": "review",
    "read": "read",
    "next": "read_next",
    "equation": "equation",
}
MODE = {
    "explain": "explain",
    "review": "review",
    "read": "reading",
    "read_next": "reading",
    "equation": "equation",
}
VERDICTS = ("correct", "partial", "incorrect")

_DONT = re.compile(
    r"\b(i\s+(still\s+)?(don'?t|do\s+not|didn'?t)\s+(get|understand|follow)|"
    r"(i'?m|i\s+am)\s+(confused|lost)|confus(ed|ing)|too\s+(hard|complex|complicated|technical)|"
    r"simpler|explain\s+(it\s+)?(again|differently)|eli5|lost\s+me)\b",
    re.IGNORECASE,
)
_HINT = re.compile(
    r"^\s*(can\s+i\s+(have|get)\s+a\s+|give\s+me\s+a\s+|another\s+|a\s+)?(hint|clue)\b", re.I
)
_SKIP = re.compile(
    r"^\s*(skip|pass|i\s+give\s+up|next\s+question|show(\s+me)?\s+the\s+answer)\b", re.I
)
_NEXT = re.compile(r"^\s*(next|continue|go\s+on|keep\s+going|next\s+section|move\s+on)\b", re.I)
_REVIEW = re.compile(r"\b(review|quiz\s+me|test\s+me|flash\s?cards?|what'?s\s+due|revise)\b", re.I)
_READ = re.compile(
    r"\b(walk\s+me\s+through|guide\s+me\s+through|read(ing)?\s+(through\s+)?(the\s+|this\s+)?paper|"
    r"let'?s\s+read)\b",
    re.I,
)
_EQUATION = re.compile(r"\b(?:equation|eq\.?)\s*\(?(\d{1,3}[a-z]?)\)?", re.I)
_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b")


def arxiv_id_in(text: str) -> str | None:
    m = _ARXIV_ID.search(text)
    return m.group(1) if m else None


def is_new_question(message: str, intent, concepts: list[str], state: dict) -> bool:
    """While a check question is pending, is this message an answer or a new question?

    Answers rarely end in '?' *and* read as a confident question of another kind about other
    concepts; a hesitant answer ("is it because the rank is low?") usually fails one of these.
    """
    if not message.strip().endswith("?"):
        return False
    if intent.intent == "chat" or intent.confidence < settings.ROUTER_MIN_CONFIDENCE:
        return False
    return set(concepts) != {state.get("concept")}


def detect(
    message: str,
    state: dict | None,
    explicit: str | None,
    args: dict | None,
    intent,
    concepts: list[str],
) -> tuple[str | None, dict]:
    """The tutor action for this turn, or None if the normal intent routing applies."""
    ts = state or {}
    pending, reading = ts.get("pending"), ts.get("reading")
    args = dict(args or {})

    if explicit:
        action = EXPLICIT.get(explicit)
        if action in ("grade", "hint") and not pending:
            return None, args  # nothing to answer or hint at: treat as a normal message
        if action == "read_next" and not reading:
            action = "skip" if pending else None
        if action == "read" and not args.get("paper_id"):
            args["paper_id"] = arxiv_id_in(message)
        if action == "equation":
            args.setdefault("paper_id", arxiv_id_in(message) or (reading or {}).get("paper_id"))
            if not args.get("equation") and (m := _EQUATION.search(message)):
                args["equation"] = m.group(1)
        return action, args

    if ts and _DONT.search(message):
        return "reexplain", args
    if pending and _HINT.match(message):
        return "hint", args
    if pending and _SKIP.match(message):
        return "skip", args
    if reading and _NEXT.match(message):
        return "read_next", args
    if _REVIEW.search(message) and len(message.split()) <= 8:
        return "review", args
    if m := _EQUATION.search(message):
        paper = arxiv_id_in(message) or (reading or {}).get("paper_id")
        if paper:
            return "equation", {**args, "paper_id": paper, "equation": m.group(1)}
    if _READ.search(message) and (paper := arxiv_id_in(message)):
        return "read", {**args, "paper_id": paper}
    if pending and not is_new_question(message, intent, concepts, ts):
        return "grade", args
    return None, args


def pitch(level_hint: str | None, offset: int) -> str:
    """Reader level (from mastery/profile) shifted down by re-explanations."""
    base = {"weak": 1, None: 2, "strong": 3}.get(level_hint, 2)
    names = (
        "simplest: everyday words, one small step at a time",
        "simple: define terms, use analogies",
        "standard: precise, assumes ML basics",
        "advanced: dense and technical",
    )
    return names[max(0, min(3, base + offset))]


def compact_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict]:
    limit = settings.TUTOR_STATE_SOURCE_CHARS
    return [{**s, "content": s["content"][:limit]} for s in sources]


def _concept(*candidates: str | None) -> str | None:
    """First candidate that maps to a vocabulary concept (mastery needs stable slugs)."""
    for c in candidates:
        if not c:
            continue
        if vocab.is_known(c):
            return c
        found = vocab.extract_concepts(c, limit=1)
        if found:
            return found[0]
    return None


def _pending(question: str | None, answer_key: str | None) -> dict | None:
    return {"question": question, "answer_key": answer_key} if question else None


def transition(
    prev: dict | None,
    action: str,
    *,
    meta: dict | None,
    text: str,
    topic: str,
    concept: str | None,
    sources: Sequence[Mapping[str, Any]],
    reading: dict | None = None,
    equation: dict | None = None,
) -> tuple[dict | None, list[dict], str | None]:
    """(new state, learning events, verdict) after the tutor's reply to `action`."""
    meta = meta or {}
    ts = dict(prev or {})
    events: list[dict] = []
    question = meta.get("question") or last_question(text)
    answer_key = meta.get("answer_key")

    if action in ("explain", "review", "read", "read_next", "equation"):
        continuing = action == "read_next"
        ts = {
            "mode": MODE[action],
            "topic": topic,
            "concept": _concept(concept, meta.get("concept")),
            "step": (ts.get("step", 0) if continuing else 0) + 1,
            "level": ts.get("level", 0) if continuing else 0,
            "hints_used": 0,
            "pending": _pending(question, answer_key),
            "sources": compact_sources(sources),
            "reading": reading,
            "equation": equation,
        }
        return ts, events, None

    if action == "skip":
        ts["pending"] = None
        return ts or None, events, None

    if action == "reexplain":
        ts.setdefault("mode", "explain")
        ts.setdefault("topic", topic)
        ts["concept"] = ts.get("concept") or _concept(concept, meta.get("concept"))
        ts["level"] = max(ts.get("level", 0) - 1, -2)
        ts["hints_used"] = 0
        ts["pending"] = _pending(question, answer_key)
        if sources:
            ts["sources"] = compact_sources(sources)
        if ts["concept"]:
            events.append(
                {"kind": "feedback", "payload": {"signal": "confused", "concepts": [ts["concept"]]}}
            )
        return ts, events, None

    pending = dict(ts.get("pending") or {})
    if action == "hint":
        ts["hints_used"] = ts.get("hints_used", 0) + 1
        if meta.get("question"):
            pending["question"] = meta["question"]
        ts["pending"] = pending or None
        return ts, events, None

    # action == "grade"
    verdict = meta.get("verdict") if meta.get("verdict") in VERDICTS else None
    if verdict and ts.get("concept"):
        events.append({"kind": "quiz", "payload": {"concept": ts["concept"], "result": verdict}})
    if verdict == "correct":
        ts["step"] = ts.get("step", 0) + 1
        ts["hints_used"] = 0
        new_question = meta.get("question") or (last_question(text) if not meta else None)
        ts["pending"] = _pending(new_question, answer_key)
    elif verdict in ("partial", "incorrect"):
        ts["hints_used"] = ts.get("hints_used", 0) + 1
        if ts["hints_used"] > settings.TUTOR_MAX_HINTS:  # the reply re-explained more simply
            ts["level"] = max(ts.get("level", 0) - 1, -2)
            ts["hints_used"] = 0
            ts["pending"] = _pending(question, answer_key) or pending or None
        elif meta.get("question"):
            ts["pending"] = {
                "question": meta["question"],
                "answer_key": answer_key or pending.get("answer_key"),
            }
    else:  # no usable verdict: keep the session, re-ask whatever the reply asked
        if question:
            ts["pending"] = {"question": question, "answer_key": pending.get("answer_key")}
    return ts, events, verdict


def is_last_step(state: dict | None) -> bool:
    return (state or {}).get("step", 0) >= settings.TUTOR_MAX_STEPS


def public_view(state: dict | None) -> dict | None:
    """What the API shows about the session (no answer keys, no source copies)."""
    if not state:
        return None
    pending = state.get("pending") or {}
    reading = state.get("reading") or {}
    return {
        "mode": state.get("mode"),
        "topic": state.get("topic"),
        "concept": state.get("concept"),
        "step": state.get("step", 0),
        "max_steps": settings.TUTOR_MAX_STEPS,
        "level": state.get("level", 0),
        "hints_used": state.get("hints_used", 0),
        "max_hints": settings.TUTOR_MAX_HINTS,
        "pending_question": pending.get("question"),
        "reading": {k: reading[k] for k in ("paper_id", "title", "index", "total", "section")}
        if reading
        else None,
    }
