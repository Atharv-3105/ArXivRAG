"""The LangGraph workflow end to end with a real LLMRouter over the offline mock."""

import pytest

from src.agents.graph import NO_SOURCES, RETRIEVAL_UNAVAILABLE, ChatDeps, build_graph
from src.core.config import settings
from src.llm.router import LLMUnavailableError
from tests.chat_fakes import (
    RETRIEVAL_DOWN,
    DeadAdapter,
    FakeRetriever,
    RecordingAdapter,
    hit,
    router,
)


def _graph(hits=None, error=None, adapter=None):
    adapter = adapter or RecordingAdapter()
    retriever = FakeRetriever(hits, error)
    return build_graph(ChatDeps(llm=router(adapter), retriever=retriever)), retriever, adapter


def _input(query, history=None):
    return {"query": query, "history": history or [], "timings": {}}


async def test_question_is_classified_retrieved_answered_and_cited():
    graph, retriever, adapter = _graph([hit("2106.09685v2", 0), hit("2305.14314v1", 0)])
    state = await graph.ainvoke(_input("Explain how LoRA works"))

    assert state["intent"] == "learn"
    assert [s["n"] for s in state["sources"]] == [1, 2]
    assert state["sources"][0]["url"] == "https://arxiv.org/html/2106.09685v2#S3.p0"
    assert state["response"] and state["citations"]  # the mock quotes and cites [1], [2]
    assert state["ungrounded"] == []
    assert state["provider"] == "mock"
    assert set(state["timings"]) == {"classify", "retrieve", "generate"}
    system = adapter.requests[0].messages[0].content
    assert "[1] LoRA injects" in system and "Answer ONLY from the numbered sources" in system
    assert adapter.requests[0].task == "tutor"  # 'learn' uses the tutor provider order


async def test_smalltalk_skips_retrieval():
    graph, retriever, adapter = _graph([hit("p", 0)])
    state = await graph.ainvoke(_input("thanks!"))
    assert retriever.queries == [] and "retrieve" not in state["timings"]
    assert len(adapter.requests) == 1


async def test_retrieval_outage_answers_safely_without_the_llm():
    graph, _, adapter = _graph(error=RETRIEVAL_DOWN)
    state = await graph.ainvoke(_input("Explain LoRA"))
    assert state["degraded"] == "retrieval_unavailable"
    assert state["response"] == RETRIEVAL_UNAVAILABLE
    assert adapter.requests == []  # nothing to ground on -> no generation


async def test_nothing_retrieved_means_no_invented_answer():
    graph, _, adapter = _graph(hits=[])
    state = await graph.ainvoke(_input("Explain the Zorblax theorem"))
    assert (state["degraded"], state["response"]) == ("no_sources", NO_SOURCES)
    assert adapter.requests == []


async def test_general_chat_without_sources_still_reaches_the_llm():
    graph, _, adapter = _graph(hits=[])
    state = await graph.ainvoke(_input("LoRA"))  # intent 'chat'
    assert state["degraded"] is None and len(adapter.requests) == 1


async def test_short_follow_up_is_searched_with_the_previous_question():
    graph, retriever, adapter = _graph([hit("p", 0)])
    history = [
        {"role": "user", "content": "Explain QLoRA quantization"},
        {"role": "assistant", "content": "QLoRA uses 4-bit NF4 [1]."},
    ]
    await graph.ainvoke(_input("and its limitations?", history))
    assert retriever.queries == ["Explain QLoRA quantization\nand its limitations?"]
    roles = [m.role for m in adapter.requests[0].messages]
    assert roles == ["system", "user", "assistant", "user"]  # history reaches the model


async def test_one_paper_cannot_fill_the_whole_context(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_PER_PAPER", 2)
    hits = [hit("big", i) for i in range(5)] + [hit("other", 0)]
    graph, _, _ = _graph(hits)
    state = await graph.ainvoke(_input("Explain LoRA"))
    assert [s["paper_id"] for s in state["sources"]] == ["big", "big", "other"]


async def test_pdf_sources_link_to_the_abstract_and_are_marked_degraded():
    pdf_hit = hit("1706.03762v7", 0, parse_method="pdf", math_fidelity="degraded")
    graph, _, adapter = _graph([pdf_hit])
    state = await graph.ainvoke(_input("Explain attention"))
    assert state["sources"][0]["url"] == "https://arxiv.org/abs/1706.03762v7"
    assert "(math: degraded)" in adapter.requests[0].messages[0].content


async def test_no_llm_available_propagates_for_the_api_to_map():
    graph, _, _ = _graph([hit("p", 0)], adapter=DeadAdapter())
    with pytest.raises(LLMUnavailableError):
        await graph.ainvoke(_input("Explain LoRA"))


async def test_stream_emits_sources_before_tokens():
    graph, _, _ = _graph([hit("p", 0)])
    kinds = [
        payload["type"]
        async for mode, payload in graph.astream(
            _input("Explain LoRA"), stream_mode=["custom", "updates"]
        )
        if mode == "custom"
    ]
    assert kinds[0] == "sources" and kinds[1] == "provider"
    assert set(kinds[2:]) == {"token"}
