"""The chat workflow (LangGraph): route -> retrieve -> one specialist (ADR 0010).

    START -> route --(plan: none)-----------------------------> specialist -> guardrail -> END
                  \\--(plan: vector|reading|equation)-> retrieve -/

    route       intent (rules, optional LLM) + tutor session commands -> agent, action, plan
    retrieve    executes the plan: hybrid search (+ graph, ADR 0009), a paper section in reading
                order, or an equation with its section
    specialist  tutor | critic | synthesizer | advisor | general (specialists.py)

Streaming: nodes push events through LangGraph's custom stream writer ({"type": "sources" |
"token" | "provider"}); under `ainvoke` the writer is a no-op, so the same graph serves /chat and
/chat/stream.

Failure policy (never block, never invent):
  retrieval fails        -> fixed "search unavailable" answer, no LLM call  (degraded)
  nothing retrieved      -> fixed "not in the indexed papers" answer, no LLM call (degraded)
  every LLM unavailable  -> LLMUnavailableError propagates; the API maps it to 503 + Retry-After
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import structlog
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.agents import guardrail, specialists
from src.agents import intent as intent_mod
from src.agents import tutor as tutor_mod
from src.agents.graph_retrieval import expand, mastery_level, rerank
from src.agents.sources import retrieval_query, select_sources, url_for
from src.agents.specialists import NO_SOURCES, RETRIEVAL_UNAVAILABLE  # noqa: F401 (re-export)
from src.agents.state import AGENTS, AgentState, Source
from src.core import metrics
from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.graph.store import GraphStore
from src.llm.router import LLMRouter
from src.memory import concepts as vocab
from src.storage.paper_content import PaperContent
from src.storage.vector_db import SearchHit

logger = structlog.get_logger(__name__)

AGENT_FOR_INTENT = {
    "learn": "tutor",
    "critique": "critic",
    "synthesize": "synthesizer",
    "recommend": "advisor",
    "chat": "general",
}


class Retriever(Protocol):
    async def search(
        self, query: str, top_k: int | None = None, paper_ids: list[str] | None = None
    ) -> list[SearchHit]: ...


@dataclass
class ChatDeps:
    llm: LLMRouter
    retriever: Retriever
    graph: GraphStore | None = None  # None: vector-only retrieval
    content: PaperContent | None = None  # None: no guided reading / equation walk-through


def _timed(name: str, fn: Callable[[AgentState], Awaitable[dict]]):
    """Plan §8: every node logs its duration and outcome; the trace id comes from contextvars."""
    try:
        from langsmith import traceable

        fn = traceable(run_type="chain", name=name)(fn)  # no-op unless LANGSMITH_TRACING
    except ImportError:  # optional dependency
        pass

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


# ================================================================== routing


def _vector(query: str, expand_as: str | None = None, **extra) -> dict:
    return {"kind": "vector", "query": query, "expand_as": expand_as, **extra}


def make_plan(agent: str, action: str | None, args: dict, state: Mapping[str, Any]) -> dict:
    """What the retrieve node should fetch for this specialist and action."""
    query = state["query"]
    history = state.get("history", [])
    ts = state.get("tutor_state") or {}
    base = retrieval_query(query, history)

    if agent == "general" and intent_mod.is_smalltalk(query):
        return {"kind": "none"}
    if agent == "tutor":
        if action in ("grade", "hint", "skip"):
            return {"kind": "none", "state_sources": True}
        if action == "reexplain":
            if ts.get("sources"):
                return {"kind": "none", "state_sources": True}
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"), query
            )
            return _vector(ts.get("topic") or last_user, "learn")
        if action == "review":
            concepts = state.get("concepts") or []
            due = state.get("due_reviews") or []
            concept = next(iter(concepts or due), None) or ts.get("concept")
            if not concept:
                return {"kind": "none", "concept": None}
            return _vector(f"{vocab.label(concept)} {concept.replace('_', ' ')}", concept=concept)
        if action == "read":
            return {"kind": "reading", "paper_id": args.get("paper_id"), "index": 0}
        if action == "read_next":
            reading = ts.get("reading") or {}
            return {
                "kind": "reading",
                "paper_id": reading.get("paper_id"),
                "index": reading.get("index", -1) + 1,
            }
        if action == "equation":
            return {
                "kind": "equation",
                "paper_id": args.get("paper_id"),
                "ref": args.get("equation"),
            }
        return _vector(base, "learn")
    if agent == "critic":
        return _vector(base, "critique")
    if agent == "synthesizer":
        return _vector(
            base,
            "synthesize",
            top_k=settings.SYNTH_TOP_K,
            max_per_paper=settings.SYNTH_MAX_PER_PAPER,
        )
    if agent == "advisor":
        concepts = state.get("concepts") or []
        if not concepts:  # "what should I read next?" has no topic: use the reader's gaps
            profile = state.get("profile") or {}
            topics = (state.get("weak_concepts") or [])[:3] + list(profile.get("interests") or [])[
                :3
            ]
            if topics:
                base = " ".join(vocab.label(t) for t in topics)
        return _vector(base, None, top_k=settings.ADVISOR_CANDIDATES, max_per_paper=1)
    return _vector(base)


# ================================================================== retrieval


async def _graph_sources(
    deps: ChatDeps, state: AgentState, query: str, hits: list[SearchHit], plan: dict
) -> list[Source]:
    """Re-rank + expand with the citation graph. Raises on any failure; the caller falls back."""
    assert deps.graph is not None
    level = mastery_level(state.get("profile"), state.get("mastery"))
    influence = await deps.graph.influence(list(dict.fromkeys(h.paper_id for h in hits)))
    top_k = plan.get("top_k") or settings.RETRIEVAL_TOP_K
    sources = select_sources(rerank(hits, level, influence), plan.get("max_per_paper"))[:top_k]
    extra = await expand(
        plan.get("expand_as") or "none", query, sources, level, deps.graph, deps.retriever
    )
    if extra:
        logger.info("Graph expansion", added=len(extra), roles=sorted({s["role"] for s in extra}))
    return sources + extra


async def _vector_retrieve(deps: ChatDeps, state: AgentState, plan: dict) -> dict:
    query = plan["query"]
    top_k = plan.get("top_k") or settings.RETRIEVAL_TOP_K
    try:
        # Over-fetch so the per-paper cap still leaves top_k sources.
        hits = await deps.retriever.search(query, top_k=top_k * 2)
    except RetrievalError as e:
        logger.error("Retrieval unavailable, answering without the LLM", error=e.message)
        return {"retrieval_query": query, "sources": [], "degraded": "retrieval_unavailable"}
    sources = select_sources(hits, plan.get("max_per_paper"))[:top_k]  # vector-only baseline
    graph_status = "off"
    if deps.graph is not None and settings.GRAPH_ENABLED and hits:
        try:
            async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
                sources = await _graph_sources(deps, state, query, hits, plan)
            graph_status = "ok"
        except Exception as e:  # plan §4: fall back to vector-only on graph failure
            graph_status = "failed"
            logger.warning("Graph retrieval failed; vector-only", error=repr(e))
    needs_sources = state.get("agent") not in ("general", "advisor")
    degraded = "no_sources" if not sources and needs_sources else None
    return {
        "retrieval_query": query,
        "sources": sources,
        "degraded": degraded,
        "graph_status": graph_status,
    }


def _section_sources(meta: dict, chunks: list[dict]) -> list[Source]:
    sources = []
    for n, c in enumerate(chunks, start=1):
        anchor = c.get("anchor")
        html = anchor and meta.get("parse_method") == "html"
        sources.append(
            Source(
                n=n,
                chunk_id=str(c["chunk_id"]),
                paper_id=meta["arxiv_id"],
                title=meta["title"],
                section=c.get("section_title") or "",
                content=c["content"],
                score=0.0,
                url=f"https://arxiv.org/html/{meta['arxiv_id']}#{anchor}"
                if html
                else f"https://arxiv.org/abs/{meta['arxiv_id']}",
                math_fidelity=meta.get("math_fidelity") or "degraded",
                role="reading",
                published=meta.get("published"),
            )
        )
    return sources


async def _reading_retrieve(deps: ChatDeps, plan: dict) -> dict:
    paper_id, content = plan.get("paper_id"), deps.content
    meta = await content.paper_meta(paper_id) if content and paper_id else None
    if content is None or meta is None:
        return {"sources": [], "degraded": "not_found", "reading": None}
    sections = await content.sections(meta["arxiv_id"])
    index = plan.get("index", 0)
    if index >= len(sections):
        return {"sources": [], "degraded": "reading_done", "reading": None}
    section = sections[index]
    chunks = await content.section_chunks(
        meta["arxiv_id"], section["idx"], settings.READING_CHUNKS_PER_SECTION
    )
    prerequisites: list[str] = []
    if index == 0 and deps.graph is not None:
        try:
            async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
                chain = await deps.graph.prerequisites(meta["arxiv_id"], 1, 3)
            prerequisites = [f"{p['title']} ({p['arxiv_id']})" for p in chain]
        except Exception as e:
            logger.warning("Prerequisites unavailable for reading", error=repr(e))
    return {
        "sources": _section_sources(meta, chunks),
        "degraded": None if chunks else "no_sources",
        "reading": {
            "paper_id": meta["arxiv_id"],
            "title": meta["title"],
            "index": index,
            "total": len(sections),
            "section": section.get("path") or section.get("title"),
            "prerequisites": prerequisites,
        },
    }


async def _equation_retrieve(deps: ChatDeps, plan: dict) -> dict:
    paper_id, ref, content = plan.get("paper_id"), plan.get("ref"), deps.content
    if content is None or not paper_id or not ref:
        return {"sources": [], "degraded": "not_found", "equation": None}
    meta = await content.paper_meta(paper_id)
    equation = await content.equation(meta["arxiv_id"], ref) if meta else None
    if meta is None or equation is None:
        return {"sources": [], "degraded": "not_found", "equation": None}
    sections = await content.sections(meta["arxiv_id"])
    path = equation.get("section_path") or ""
    section = next(
        (
            s
            for s in sections
            if path and (s.get("path") == path or path.endswith(s["title"] or ""))
        ),
        None,
    )
    chunks = (
        await content.section_chunks(
            meta["arxiv_id"], section["idx"], settings.READING_CHUNKS_PER_SECTION
        )
        if section
        else []
    )
    return {
        "sources": _section_sources(meta, chunks),
        "degraded": None,
        "equation": {
            "paper_id": meta["arxiv_id"],
            "title": meta["title"],
            "number": equation.get("number"),
            "latex": equation["latex"],
            "section": path,
        },
    }


# ================================================================== input gate


def _gated(reason: str, flags: dict) -> dict:
    metrics.INPUT_GATE.labels(reason).inc()
    return {
        "intent": "chat",
        "intent_confidence": 1.0,
        "agent": "general",
        "tutor_action": None,
        "plan": {"kind": "none"},
        "degraded": reason,
        "input_flags": flags,
    }


async def _moderation_gate(deps: ChatDeps, state: AgentState, flags: dict) -> dict | None:
    """ADR 0011: optional LLM moderation before anything else. None = let the turn through."""
    if not (settings.GUARDRAIL_ENABLED and settings.GUARDRAIL_MODERATION):
        return None
    flags["moderation"] = await guardrail.moderate(state["query"], deps.llm)
    if not flags["moderation"]["allowed"]:
        logger.warning("Input refused by moderation", category=flags["moderation"]["category"])
        return _gated("unsafe", flags)
    return None


def _out_of_domain(state: AgentState) -> bool:
    if not (settings.GUARDRAIL_ENABLED and settings.GUARDRAIL_DOMAIN_GATE):
        return False
    return not guardrail.in_domain(
        state["query"], state.get("history", []), state.get("tutor_state"), state.get("action")
    )


# ================================================================== the graph


def build_graph(deps: ChatDeps):
    async def route(state: AgentState) -> dict:
        query = state["query"]
        flags: dict = {"prompt_injection": guardrail.prompt_injection(query)}
        if flags["prompt_injection"]:
            logger.warning("Possible prompt injection", preview=query[:120])
        refused = await _moderation_gate(deps, state, flags)
        if refused is not None:
            return refused
        result = await intent_mod.classify(query, deps.llm)
        action, args = tutor_mod.detect(
            query,
            state.get("tutor_state"),
            state.get("action"),
            state.get("action_args"),
            result,
            state.get("concepts") or [],
        )
        # Tutor commands ("quiz me", "next", "hint") are in scope by definition; only free-form
        # messages go through the domain gate.
        if action is None and _out_of_domain(state):
            logger.info("Out-of-domain question refused", preview=query[:80])
            return _gated("out_of_domain", flags)
        metrics.INPUT_GATE.labels("pass").inc()
        if action:
            agent = "tutor"
        else:
            agent = AGENT_FOR_INTENT[result.intent]
            action = "explain" if agent == "tutor" else None
        plan = make_plan(agent, action, args, {**state, "agent": agent})
        logger.info(
            "Routed",
            intent=result.intent,
            confidence=result.confidence,
            method=result.method,
            agent=agent,
            action=action,
            plan=plan["kind"],
        )
        update: dict = {
            "intent": result.intent,
            "intent_confidence": result.confidence,
            "agent": agent,
            "tutor_action": action,
            "plan": plan,
            "input_flags": flags,
        }
        if plan.get("state_sources"):
            update["sources"] = (state.get("tutor_state") or {}).get("sources", [])
        return update

    async def retrieve(state: AgentState) -> dict:
        plan = state["plan"]
        if plan["kind"] == "reading":
            update = await _reading_retrieve(deps, plan)
        elif plan["kind"] == "equation":
            update = await _equation_retrieve(deps, plan)
        else:
            update = await _vector_retrieve(deps, state, plan)
        get_stream_writer()({"type": "sources", "sources": update.get("sources", [])})
        return update

    async def guard(state: AgentState) -> dict:
        """ADR 0011: check the finished answer; append a visible notice if it is not approved."""
        if not settings.GUARDRAIL_ENABLED or state.get("provider") in ("none", "advisor"):
            decision = guardrail.GuardDecision(True, 1.0, metrics={"skipped": True})
        else:
            lenient = state.get("agent") == "tutor" and state.get("tutor_action") in (
                "grade",
                "hint",
            )
            decision = guardrail.check_answer(
                state.get("response", ""),
                state.get("sources", []),
                state.get("ungrounded", []),
                strict=not lenient,
            )
        for issue in decision.issues or ["none"]:
            metrics.GUARDRAIL.labels(str(decision.approved).lower(), issue).inc()
        update: dict = {"guardrail": {**decision.to_dict(), "input": state.get("input_flags", {})}}
        if decision.notice:
            logger.warning("Guardrail flagged answer", issues=decision.issues)
            get_stream_writer()({"type": "token", "text": decision.notice})
            update["response"] = state.get("response", "") + decision.notice
        return update

    def after_route(state: AgentState) -> str:
        return state["agent"] if state["plan"]["kind"] == "none" else "retrieve"

    def after_retrieve(state: AgentState) -> str:
        return state["agent"]

    nodes = {
        "tutor": specialists.make_tutor(deps),
        "critic": specialists.make_critic(deps),
        "synthesizer": specialists.make_synthesizer(deps),
        "advisor": specialists.make_advisor(deps),
        "general": specialists.make_general(deps),
    }
    graph = StateGraph(AgentState)
    graph.add_node("route", _timed("route", route))
    graph.add_node("retrieve", _timed("retrieve", retrieve))
    graph.add_node("guardrail", _timed("guardrail", guard))
    graph.add_edge("guardrail", END)
    for name in AGENTS:
        graph.add_node(name, _timed(name, nodes[name]))
        graph.add_edge(name, "guardrail")
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ["retrieve", *AGENTS])
    graph.add_conditional_edges("retrieve", after_retrieve, list(AGENTS))
    return graph.compile()


__all__ = ["ChatDeps", "Retriever", "build_graph", "select_sources", "url_for", "make_plan"]
