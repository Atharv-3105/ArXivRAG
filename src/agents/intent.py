"""Intent classification: which behaviour the answer should have.

Rules first, because they are free, instant and deterministic, and on a free tier every LLM call
spends quota the answer itself needs. An LLM classifier is opt-in (ROUTER_USE_LLM) and falls back
to the rules on any failure. Low confidence always means "chat" (plan §4): a wrong specialist
(e.g. a critique when the user wanted an explanation) is worse than a general answer.
"""

import json
import re
from dataclasses import dataclass

import structlog

from src.agents.state import INTENTS, Intent
from src.core.config import settings
from src.llm.base import ChatMessage, LLMRequest
from src.llm.router import LLMRouter

logger = structlog.get_logger(__name__)

_RULES: dict[Intent, list[str]] = {
    "critique": [
        r"\bcritique\b",
        r"\bcriticis|criticiz",
        r"\bweakness",
        r"\blimitation",
        r"\bflaw",
        r"\bassumption",
        r"\bbaseline",
        r"\breproduc",
        r"\bvalid(ity)?\b",
        r"\bsound\b",
        r"\bshortcoming",
        r"\bconfound",
        r"\bablation",
    ],
    "synthesize": [
        r"\bcompar",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference(s)? between\b",
        r"\btrend",
        r"\bsurvey\b",
        r"\blandscape\b",
        r"\bacross (papers|methods|works)\b",
        r"\bconsensus\b",
        r"\bcontradict",
        r"\bstate of the art\b",
        r"\bsota\b",
        r"\bhow do .+ differ\b",
    ],
    "recommend": [
        r"\brecommend",
        r"\bwhat (should|do) i read\b",
        r"\bnext paper",
        r"\bsuggest",
        r"\breading list\b",
        r"\bwhere (should i|to) start\b",
        r"\blearning path\b",
        r"\bwhich papers?\b",
        r"\bprerequisite",
    ],
    "learn": [
        r"\bexplain",
        r"\bwhat (is|are|does)\b",
        r"\bhow (does|do|is|can)\b",
        r"\bwhy (does|do|is)\b",
        r"\bteach\b",
        r"\bunderstand",
        r"\bintuition",
        r"\bderiv",
        r"\bwalk me through\b",
        r"\beli5\b",
        r"\bstep by step\b",
        r"\bmeaning of\b",
        r"\bdefine\b",
    ],
}
_COMPILED = {intent: [re.compile(p, re.IGNORECASE) for p in ps] for intent, ps in _RULES.items()}
_SMALLTALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thx|ok(ay)?|cool|great|bye|good (morning|evening))"
    r"[\s!.?]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    method: str  # "rules" | "llm" | "smalltalk"


def is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK.match(text))


def classify_rules(text: str) -> IntentResult:
    if is_smalltalk(text):
        return IntentResult("chat", 1.0, "smalltalk")
    scores = {i: sum(1 for p in ps if p.search(text)) for i, ps in _COMPILED.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top == 0:
        return IntentResult("chat", 0.3, "rules")
    # Margin-based confidence: one clear winner -> high; a tie -> 0.5 (ambiguous).
    confidence = round(top / (top + second), 2)
    if confidence < settings.ROUTER_MIN_CONFIDENCE:
        return IntentResult("chat", confidence, "rules")
    return IntentResult(best, confidence, "rules")


_LLM_PROMPT = (
    "Classify the user's message about research papers into exactly one intent:\n"
    "learn (wants an explanation), critique (wants weaknesses/validity assessed), synthesize "
    "(compare or survey several works), recommend (what to read next), chat (anything else).\n"
    'Reply with JSON only: {"intent": "<one of the five>", "confidence": <0..1>}'
)


def _parse_llm(text: str) -> IntentResult | None:
    try:
        data = json.loads(text)
        intent, confidence = data["intent"], float(data["confidence"])
    except (ValueError, KeyError, TypeError):
        return None
    if intent not in INTENTS or not 0 <= confidence <= 1:
        return None
    return IntentResult(intent, confidence, "llm")


async def classify(text: str, llm: LLMRouter | None = None) -> IntentResult:
    rules = classify_rules(text)
    if not settings.ROUTER_USE_LLM or llm is None or rules.method == "smalltalk":
        return rules
    request = LLMRequest(
        messages=[ChatMessage("system", _LLM_PROMPT), ChatMessage("user", text[:2000])],
        task="router",
        max_tokens=40,
        temperature=0.0,
        json_mode=True,
    )
    try:
        result = await llm.complete(request, validate=lambda t: _parse_llm(t) is not None)
    except Exception as e:  # any LLM trouble: the rules answer is always available
        logger.warning("LLM intent classification failed, using rules", error=str(e))
        return rules
    parsed = _parse_llm(result.text)
    if parsed is None or parsed.confidence < settings.ROUTER_MIN_CONFIDENCE:
        return IntentResult("chat", parsed.confidence if parsed else 0.0, "llm")
    return parsed
