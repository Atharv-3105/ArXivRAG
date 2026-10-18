"""Guardrail: input gate before any LLM call, grounding checks after every answer (ADR 0011).

Input (in the route node, before retrieval or generation):
  domain gate      clearly non-ML questions get a fixed reply: no quota spent on "write a poem",
                   and no LLM improvising from whatever the vector index returned (vector search
                   always returns *something*)
  injection flag   "ignore previous instructions" etc. is logged and recorded; the grounding rules
                   stay in the system prompt either way
  moderation       optional LLM classifier (GUARDRAIL_MODERATION, off by default: it costs a call
                   per message on a free quota)

Output (a node after every specialist):
  ungrounded       [n] that matches no retrieved source                         -> blocks
  uncited          share of substantive sentences with no citation              -> blocks
  weak support     cited sentence shares too few content words with its source  -> blocks
  degraded math    a formula copied from a PDF-extracted source                 -> blocks
  malformed math   unbalanced braces or dollar signs                            -> recorded

The answer has already been streamed, so "block" cannot mean "delete". It means: not approved,
and a visible correction notice is appended telling the reader what to double-check. Measured
rates per template/model feed the Phase 6 eval, where prompts and models get fixed.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from src.agents import intent as intent_mod
from src.agents.checks import _CITATION
from src.agents.grounding import cited_numbers
from src.core.config import settings
from src.memory import concepts as vocab

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------ input

_INJECTION = re.compile(
    r"(ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)|"
    r"disregard\s+(the\s+|your\s+)?(system|previous|above)|you\s+are\s+now\s+|"
    r"pretend\s+(to\s+be|you\s+are)|(reveal|print|show)\s+(me\s+)?(your|the)\s+(system\s+)?prompt|"
    r"jailbreak|developer\s+mode|do\s+anything\s+now)",
    re.IGNORECASE,
)

# Words that make a message plausibly about ML research. Deliberately generous: a false refusal
# (an ML question refused) is worse than a false pass (the grounded pipeline then finds nothing).
ML_LEXICON = frozenset(
    """
    ai ml nlp cv rl llm llms gpt bert model models modelling modeling learning learn learned
    neural network networks deep training train trained pretraining finetune finetuning
    fine-tuning inference dataset datasets data benchmark benchmarks baseline baselines
    evaluation metric metrics accuracy loss objective gradient gradients optimizer optimization
    optimisation parameter parameters weights layer layers embedding embeddings vector vectors
    token tokens tokenizer attention transformer transformers encoder decoder architecture
    convolution recurrent generative diffusion classifier classification regression clustering
    supervised unsupervised self-supervised reinforcement reward policy agent agents prompt
    prompting retrieval rag sampling decoding probability probabilistic bayesian likelihood
    distribution variance bias overfitting regularization generalization generalisation
    scaling compute gpu tpu flops quantization pruning distillation paper papers arxiv
    preprint citation citations author authors abstract method methods approach experiment
    experiments ablation results theorem proof lemma equation equations formula derivation
    matrix matrices tensor linear nonlinear algorithm algorithms complexity convergence
    hyperparameter hyperparameters epoch epochs batch vision language speech multimodal
    robotics graph graphs kernel softmax logits entropy perplexity alignment hallucination
    research study studies survey section figure table appendix contribution contributions
    read reading bayes statistics statistical math mathematics calculus derivative integral
    eigenvalue eigenvector expectation estimator estimation sample samples hypothesis
    prefill context latency throughput cache sparse sparsity efficient efficiency lora qlora
    adapter adapters rlhf dpo ppo moe transformer-xl mamba rnn lstm cnn gan vae sft
    reasoning chain-of-thought cot rerank reranking ranker retriever benchmarking
    """.split()
)
_WORD = re.compile(r"[a-zA-Z][a-zA-Z\-]+")
# "thanks, that helped" / "ok got it": acknowledgements belong to the conversation, not a topic.
_ACK = re.compile(r"^\s*(thanks|thank\s+you|thx|ok(ay)?|got\s+it|great|cool|nice|perfect)\b", re.I)
_ARXIV_ID = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b|\b[a-z\-]+/\d{7}\b")


def prompt_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def has_ml_signal(text: str) -> bool:
    if vocab.extract_concepts(text, limit=1) or _ARXIV_ID.search(text):
        return True
    words: set[str] = set()
    for w in _WORD.findall(text):
        w = w.lower().strip("-")
        words.add(w)
        words.update(part for part in w.split("-") if part)  # "RBS-Attention" -> "attention"
    return bool(words & ML_LEXICON)


def in_domain(
    query: str, history: list[dict], tutor_state: dict | None, action: str | None
) -> bool:
    """False only when nothing points to ML research: not the message, not the recent user
    turns (follow-ups like "why?" inherit their topic), not an active tutoring session."""
    if action or tutor_state or intent_mod.is_smalltalk(query) or _ACK.match(query):
        return True
    if has_ml_signal(query):
        return True
    recent = [m["content"] for m in history if m.get("role") == "user"][-3:]
    return any(has_ml_signal(m) for m in recent)


_MODERATION_PROMPT = (
    "You are a content-safety classifier for an AI research tutoring app. Decide whether the "
    "user message asks for clearly harmful content (violence, weapons, self-harm instructions, "
    "sexual content involving minors, targeted harassment, malware). Research discussion of AI "
    "safety topics is allowed. Reply with JSON only: "
    '{"allowed": true|false, "category": "<short label or none>"}'
)


def _parse_moderation(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) and isinstance(data.get("allowed"), bool) else None


async def moderate(query: str, llm) -> dict:
    """{"allowed": bool, "category": str}. Fails open (allowed) on any error: the grounded,
    domain-gated pipeline is the primary defence and moderation must not take the product down."""
    from src.llm.base import ChatMessage, LLMRequest

    request = LLMRequest(
        messages=[ChatMessage("system", _MODERATION_PROMPT), ChatMessage("user", query[:2000])],
        task="router",
        max_tokens=30,
        temperature=0.0,
        json_mode=True,
        tags={"agent": "guardrail"},
    )
    try:
        result = await llm.complete(request, validate=lambda t: _parse_moderation(t) is not None)
        return _parse_moderation(result.text) or {"allowed": True, "category": "unparsed"}
    except Exception as e:
        logger.warning("Moderation unavailable, allowing", error=str(e))
        return {"allowed": True, "category": "unavailable"}


# ------------------------------------------------------------------ output

_MATH = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.DOTALL)
_CONTENT_WORD = re.compile(r"[a-z][a-z0-9\-]{3,}")
_STOP = frozenset(
    "this that these those with from have been were which their there they them into over than "
    "such also only more most some other about when where what while because between through "
    "each both then very will would could should does used using based show shows shown".split()
)
ISSUE_TEXT = {
    "ungrounded_citations": "it cites sources that were not retrieved",
    "uncited_claims": "several statements carry no citation",
    "weak_support": "some cited passages do not clearly support the statement",
    "degraded_math_quoted": "a formula comes from a PDF-extracted source and may be garbled",
}
BLOCKING = frozenset(ISSUE_TEXT)


@dataclass
class GuardDecision:
    approved: bool
    confidence: float
    issues: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    notice: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("notice")
        return data


def _content_words(text: str) -> set[str]:
    return {w for w in _CONTENT_WORD.findall(text.lower()) if w not in _STOP}


def _normalise_latex(latex: str) -> str:
    return re.sub(r"\s+", "", latex)


def _balanced(latex: str) -> bool:
    depth = 0
    for i, ch in enumerate(latex):
        if ch == "{" and (i == 0 or latex[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or latex[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def substantive_sentences(text: str) -> list[str]:
    """Claims worth checking: not headings, table rows, questions, or short fragments."""
    body = _MATH.sub(" MATH ", text)
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|")) or set(line) <= set("-|: "):
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line)
        for sentence in re.split(r"(?<=[.!])\s+", line):
            s = sentence.strip()
            if len(s.split()) >= 6 and not s.endswith("?") and not s.startswith("("):
                out.append(s)
    return out


def check_answer(
    text: str,
    sources: Sequence[Mapping[str, Any]],
    ungrounded: list[int],
    strict: bool = True,
) -> GuardDecision:
    """strict=False (tutor feedback, hints) skips the citation-coverage checks: a two-line
    "that's right, now consider..." does not need a citation per sentence."""
    by_n = {s["n"]: s for s in sources}
    issues: list[str] = []
    metrics: dict = {}

    if ungrounded:
        issues.append("ungrounded_citations")
    metrics["ungrounded"] = len(ungrounded)

    sentences = substantive_sentences(text)
    cited = [s for s in sentences if _CITATION.search(s)]
    metrics["sentences"] = len(sentences)
    metrics["cited_share"] = round(len(cited) / len(sentences), 3) if sentences else 1.0
    if (
        strict
        and len(sentences) >= 3
        and 1 - metrics["cited_share"] > settings.GUARDRAIL_MAX_UNCITED
    ):
        issues.append("uncited_claims")

    weak = 0
    for sentence in cited:
        words = _content_words(_CITATION.sub(" ", sentence))
        source_words: set[str] = set()
        for n in cited_numbers(sentence):
            if n in by_n:
                source_words |= _content_words(by_n[n]["content"] + " " + by_n[n]["title"])
        if (
            words
            and source_words
            and len(words & source_words) / len(words) < settings.GUARDRAIL_MIN_SUPPORT
        ):
            weak += 1
    metrics["weakly_supported"] = weak
    if strict and cited and weak / len(cited) > 0.5:
        issues.append("weak_support")

    formulas = [a or b for a, b in _MATH.findall(text)]
    malformed = [f for f in formulas if not _balanced(f)]
    grounded_formulas = degraded = 0
    for f in formulas:
        norm = _normalise_latex(f)
        if len(norm) < 4:
            continue
        for s in sources:
            if norm in _normalise_latex(s["content"]):
                grounded_formulas += 1
                if s.get("math_fidelity") == "degraded":
                    degraded += 1
                break
    metrics.update(
        formulas=len(formulas), formulas_grounded=grounded_formulas, malformed=len(malformed)
    )
    if degraded:
        issues.append("degraded_math_quoted")
    if malformed:
        issues.append("malformed_math")

    blocking = [i for i in issues if i in BLOCKING]
    penalty = 0.4 * bool(ungrounded) + 0.25 * ("uncited_claims" in issues)
    penalty += 0.25 * ("weak_support" in issues) + 0.2 * bool(degraded) + 0.1 * bool(malformed)
    decision = GuardDecision(
        approved=not blocking,
        confidence=round(max(0.0, 1.0 - penalty), 2),
        issues=issues,
        metrics=metrics,
    )
    if blocking:
        reasons = "; ".join(ISSUE_TEXT[i] for i in blocking)
        decision.notice = f"\n\n> **Check before relying on this:** {reasons}."
    return decision
