import json

import pytest

from src.agents import intent
from src.agents.grounding import cited_numbers, resolve_citations
from src.core.config import settings
from tests.chat_fakes import DeadAdapter, RecordingAdapter, router


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Explain how LoRA reduces the number of trainable parameters", "learn"),
        ("What is rotary position embedding?", "learn"),
        ("What are the weaknesses of this paper's baselines?", "critique"),
        ("Is the evaluation reproducible? Critique the ablation.", "critique"),
        ("Compare FlashAttention and PagedAttention", "synthesize"),
        ("RLHF vs DPO: how do they differ", "synthesize"),
        ("What should I read next after the transformer paper?", "recommend"),
        ("Recommend papers on mixture of experts", "recommend"),
        ("LoRA", "chat"),  # no signal: low confidence -> chat
        ("thanks!", "chat"),
    ],
)
def test_rules_classify_intents(text, expected):
    assert intent.classify_rules(text).intent == expected


def test_ambiguous_message_falls_back_to_chat():
    result = intent.classify_rules("Explain the limitations")  # learn and critique tie
    assert (result.intent, result.confidence) == ("chat", 0.5)


async def test_llm_classifier_is_used_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "ROUTER_USE_LLM", True)
    adapter = RecordingAdapter()
    adapter._answer = lambda req: json.dumps({"intent": "critique", "confidence": 0.9})
    result = await intent.classify("is this sound?", router(adapter))
    assert (result.intent, result.method) == ("critique", "llm")
    assert adapter.requests[0].task == "router" and adapter.requests[0].json_mode


async def test_llm_classifier_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(settings, "ROUTER_USE_LLM", True)
    result = await intent.classify("Compare BERT and GPT", router(DeadAdapter()))
    assert (result.intent, result.method) == ("synthesize", "rules")


async def test_llm_classifier_is_off_by_default():
    adapter = RecordingAdapter()
    await intent.classify("Explain attention", router(adapter))
    assert adapter.requests == []  # no quota spent


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("A [1] and B [2].", [1, 2]),
        ("Both [1, 3] agree [1][2].", [1, 3, 2]),
        ("See [2-4].", [2, 3, 4]),
        ("No citations, but an array a[0], x_[1] and [x].", []),
    ],
)
def test_cited_numbers(text, numbers):
    assert cited_numbers(text) == numbers


def test_citations_to_unretrieved_sources_are_flagged():
    sources = [
        {"n": 1, "paper_id": "p1", "title": "T1", "section": "S", "url": "u1"},
        {"n": 2, "paper_id": "p2", "title": "T2", "section": "S", "url": "u2"},
    ]
    citations, ungrounded = resolve_citations("x [2] y [7] z [1]", sources)
    assert [c["n"] for c in citations] == [2, 1]
    assert ungrounded == [7]
