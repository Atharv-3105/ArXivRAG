"""The chat workflow: classify -> retrieve -> generate (LangGraph).

    START -> classify --(smalltalk)--------------------> generate -> END
                      \\--(anything else)-> retrieve -/

Streaming: nodes push events through LangGraph's custom stream writer ({"type": "sources" |
"token" | "provider"}). Under `astream(stream_mode="custom")` they reach the SSE endpoint; under
`ainvoke` the writer is a no-op, so the same graph serves both /chat and /chat/stream.

Failure policy (never block, never invent):
  retrieval fails        -> fixed "search unavailable" answer, no LLM call  (degraded)
  nothing retrieved      -> fixed "not in the indexed papers" answer, no LLM call (degraded)
                            except smalltalk/chat, which needs no sources
  every LLM unavailable  -> LLMUnavailableError propagates; the API maps it to 503 + Retry-After
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import structlog
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.agents import intent as intent_mod
from src.agents.grounding import resolve_citations
from src.agents.prompts import PROMPT_VERSION, build_messages
from src.agents.state import AgentState, Source
from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.llm.base import LLMRequest
from src.llm.router import LLMRouter
from src.storage.vector_db import SearchHit

logger = structlog.get_logger(__name__)

RETRIEVAL_UNAVAILABLE = (
    "Paper search is temporarily unavailable, so I can't give you a grounded answer right now. "
    "Please try again in a minute."
)
NO_SOURCES = (
    "I couldn't find anything about this in the indexed papers. Try rephrasing with the "
    "method or paper name, or ask for papers on this topic to be ingested."
)


class Retriever(Protocol):
    async def search(
        self, query: str, top_k: int | None = None, paper_ids: list[str] | None = None
    ) -> list[SearchHit]: ...


@dataclass
class ChatDeps:
    llm: LLMRouter
    retriever: Retriever


def _traceable(fn):
    """LangSmith span when LANGSMITH_TRACING is on; langsmith makes this a no-op otherwise."""
    try:
        from langsmith import traceable
    except ImportError:  # optional dependency
        return fn
    return traceable(run_type="chain", name=fn.__name__)(fn)


def _timed(name: str, fn: Callable[[AgentState], Awaitable[dict]]):
    """Plan §8: every node logs its duration and outcome; the trace id comes from contextvars."""

    async def node(state: AgentState) -> dict:
        started = time.perf_counter()
        try:
            update = await fn(state)
        except Exception as e:
            logger.error("Node failed", node=name, error=str(e), error_type=type(e).__name__)
            raise
        ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info("Node finished", node=name, duration_ms=ms, keys=sorted(update))
        return {**update, "timings": {name: ms}}

    node.__name__ = name
    return node


def retrieval_query(query: str, history: list[dict]) -> str:
    """Short follow-ups ("and its limitations?") are searched together with the previous user
    message. An LLM rewrite would be better but costs a call per turn on a free quota."""
    if len(query.split()) > settings.FOLLOWUP_MAX_WORDS:
        return query
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    return f"{previous}\n{query}" if previous else query


def _url(hit: SearchHit) -> str:
    anchor = hit.metadata.get("anchor")
    if anchor and hit.metadata.get("parse_method") == "html":
        return f"https://arxiv.org/html/{hit.paper_id}#{anchor}"
    return f"https://arxiv.org/abs/{hit.paper_id}"


def select_sources(hits: list[SearchHit]) -> list[Source]:
    """Keep ranking order, cap chunks per paper so one long paper cannot fill the context."""
    per_paper: dict[str, int] = {}
    sources: list[Source] = []
    for hit in hits:
        if per_paper.get(hit.paper_id, 0) >= settings.RETRIEVAL_MAX_PER_PAPER:
            continue
        per_paper[hit.paper_id] = per_paper.get(hit.paper_id, 0) + 1
        sources.append(
            Source(
                n=len(sources) + 1,
                chunk_id=hit.chunk_id,
                paper_id=hit.paper_id,
                title=hit.title,
                section=hit.section,
                content=hit.content,
                score=round(hit.score, 4),
                url=_url(hit),
                math_fidelity=hit.metadata.get("math_fidelity", "degraded"),
            )
        )
    return sources


def build_graph(deps: ChatDeps):
    async def classify(state: AgentState) -> dict:
        result = await intent_mod.classify(state["query"], deps.llm)
        logger.info(
            "Intent classified",
            intent=result.intent,
            confidence=result.confidence,
            method=result.method,
        )
        return {"intent": result.intent, "intent_confidence": result.confidence}

    @_traceable
    async def retrieve(state: AgentState) -> dict:
        writer = get_stream_writer()
        query = retrieval_query(state["query"], state.get("history", []))
        try:
            # Over-fetch so the per-paper cap still leaves RETRIEVAL_TOP_K sources.
            hits = await deps.retriever.search(query, top_k=settings.RETRIEVAL_TOP_K * 2)
        except RetrievalError as e:
            logger.error("Retrieval unavailable, answering without the LLM", error=e.message)
            return {"retrieval_query": query, "sources": [], "degraded": "retrieval_unavailable"}
        sources = select_sources(hits)[: settings.RETRIEVAL_TOP_K]
        writer({"type": "sources", "sources": sources})
        degraded = "no_sources" if not sources and state.get("intent") != "chat" else None
        return {"retrieval_query": query, "sources": sources, "degraded": degraded}

    @_traceable
    async def generate(state: AgentState) -> dict:
        writer = get_stream_writer()
        degraded = state.get("degraded")
        if degraded:
            text = RETRIEVAL_UNAVAILABLE if degraded == "retrieval_unavailable" else NO_SOURCES
            writer({"type": "token", "text": text})
            return {"response": text, "citations": [], "ungrounded": [], "provider": "none"}

        sources = state.get("sources", [])
        history = state.get("history", [])[-settings.CHAT_HISTORY_MESSAGES :]
        request = LLMRequest(
            messages=build_messages(state["query"], state["intent"], sources, history),
            task="tutor" if state["intent"] == "learn" else "default",
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        parts: list[str] = []
        provider = model = ""
        async for event in deps.llm.stream(request):
            if event.kind == "start":
                provider, model = event.provider, event.model
                writer({"type": "provider", "provider": provider, "model": model})
            elif event.kind == "token":
                parts.append(event.text)
                writer({"type": "token", "text": event.text})
        text = "".join(parts)
        citations, ungrounded = resolve_citations(text, sources)
        if ungrounded:
            logger.warning("Answer cites sources that were not retrieved", numbers=ungrounded)
        logger.info(
            "Answer generated",
            provider=provider,
            prompt_version=PROMPT_VERSION,
            sources=len(sources),
            citations=len(citations),
        )
        return {
            "response": text,
            "citations": citations,
            "ungrounded": ungrounded,
            "provider": provider,
            "model": model,
        }

    def after_classify(state: AgentState) -> str:
        return "generate" if intent_mod.is_smalltalk(state["query"]) else "retrieve"

    graph = StateGraph(AgentState)
    graph.add_node("classify", _timed("classify", classify))
    graph.add_node("retrieve", _timed("retrieve", retrieve))
    graph.add_node("generate", _timed("generate", generate))
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", after_classify, ["retrieve", "generate"])
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
