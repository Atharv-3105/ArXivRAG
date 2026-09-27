"""The specialist nodes: Tutor, Critic, Synthesizer, Advisor, General (ADR 0010).

Each specialist is a LangGraph node with its own prompt template (versioned, A/B-able), output
contract and checks - not an autonomous agent. One LLM call per turn at most (the Advisor makes
none), so a turn costs the same quota whichever specialist answers.
"""

import asyncio
from dataclasses import dataclass

import structlog
from langgraph.config import get_stream_writer

from src.agents import advisor as advisor_mod
from src.agents import checks as checks_mod
from src.agents import prompts
from src.agents import tutor as tutor_mod
from src.agents.graph_retrieval import mastery_level
from src.agents.grounding import resolve_citations
from src.agents.state import AgentState, Source
from src.agents.trailer import TrailerSplitter, parse_meta
from src.core.config import settings
from src.llm.base import LLMRequest
from src.memory import concepts as vocab

logger = structlog.get_logger(__name__)

RETRIEVAL_UNAVAILABLE = (
    "Paper search is temporarily unavailable, so I can't give you a grounded answer right now. "
    "Please try again in a minute."
)
NO_SOURCES = (
    "I couldn't find anything about this in the indexed papers. Try rephrasing with the "
    "method or paper name, or ask for papers on this topic to be ingested."
)
NOT_FOUND = (
    "I couldn't find that paper or equation in the indexed corpus. Check the arXiv id (and the "
    "equation number), or ask for the paper to be ingested."
)
NOTHING_DUE = (
    'Nothing is due for review right now. Ask me to explain a concept, or say "quiz me on '
    '<concept>" to practise one anyway.'
)
READING_DONE = (
    'That was the last section of the paper. Say "quiz me" to review what you learned, or ask '
    "me to compare it with related work."
)
OUT_OF_DOMAIN = (
    "I'm a research assistant for AI and machine-learning papers, so I can't help with that. "
    "Ask me about a model, method, paper or concept, and I'll answer from the indexed papers."
)
UNSAFE = "I can't help with that request."
_DEGRADED = {
    "out_of_domain": OUT_OF_DOMAIN,
    "unsafe": UNSAFE,
    "retrieval_unavailable": RETRIEVAL_UNAVAILABLE,
    "no_sources": NO_SOURCES,
    "not_found": NOT_FOUND,
}


def _degraded_text(state: AgentState) -> str:
    return _DEGRADED.get(state.get("degraded") or "", NO_SOURCES)


TASK = {"tutor": "tutor", "critic": "critic", "synthesizer": "synthesizer", "general": "default"}


@dataclass
class Generated:
    text: str
    meta: dict | None
    provider: str
    model: str


async def generate(deps, system: str, sources, history, user_message, *, agent, tags, trailer):
    """One streamed LLM call. With `trailer`, text after <<<META>>> is parsed, never shown."""
    writer = get_stream_writer()
    request = LLMRequest(
        messages=prompts.build_messages(system, sources, history, user_message),
        task=TASK[agent],
        max_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
        tags={"agent": agent, **tags},
    )
    splitter = TrailerSplitter() if trailer else None
    parts: list[str] = []
    provider = model = ""
    async for event in deps.llm.stream(request):
        if event.kind == "start":
            provider, model = event.provider, event.model
            writer({"type": "provider", "provider": provider, "model": model})
        elif event.kind == "token":
            visible = splitter.feed(event.text) if splitter else event.text
            if visible:
                parts.append(visible)
                writer({"type": "token", "text": visible})
    if splitter:
        tail = splitter.close()
        if tail:
            parts.append(tail)
            writer({"type": "token", "text": tail})
    meta = parse_meta(splitter.meta_text) if splitter and splitter.in_meta else None
    return Generated("".join(parts), meta, provider, model)


def _history(state: AgentState) -> list[dict]:
    return state.get("history", [])[-settings.CHAT_HISTORY_MESSAGES :]


def fixed(text: str, **extra) -> dict:
    """A reply that needs no LLM (degraded paths, skip, nothing due, the Advisor)."""
    get_stream_writer()({"type": "token", "text": text})
    return {
        "response": text,
        "citations": [],
        "ungrounded": [],
        "provider": "none",
        "model": "",
        "prompt_version": None,
        "checks": {},
        **extra,
    }


def finish(agent: str, text: str, sources: list[Source], gen, version, checks, **extra) -> dict:
    citations, ungrounded = resolve_citations(text, sources)
    checks = {**checks, "citations_grounded": not ungrounded}
    failed = sorted(k for k, ok in checks.items() if not ok)
    if ungrounded:
        logger.warning("Answer cites sources that were not retrieved", numbers=ungrounded)
    logger.info(
        "Specialist answered",
        agent=agent,
        prompt_version=version,
        provider=gen.provider if gen else "none",
        sources=len(sources),
        citations=len(citations),
        failed_checks=failed,
    )
    return {
        "response": text,
        "citations": citations,
        "ungrounded": ungrounded,
        "provider": gen.provider if gen else "none",
        "model": gen.model if gen else "",
        "prompt_version": version,
        "checks": checks,
        **extra,
    }


def _reader(state: AgentState) -> list[str]:
    return prompts.reader_lines(state.get("profile"), state.get("mastery"))


# ================================================================== General


def make_general(deps):
    async def general(state: AgentState) -> dict:
        if state.get("degraded"):
            return fixed(_degraded_text(state))
        sources = state.get("sources", [])
        system, version = prompts.render("general", state.get("user_id"), reader=_reader(state))
        gen = await generate(
            deps,
            system,
            sources,
            _history(state),
            state["query"],
            agent="general",
            tags={"prompt": version},
            trailer=False,
        )
        return finish("general", gen.text, sources, gen, version, {})

    return general


# ================================================================== Critic


def make_critic(deps):
    async def critic(state: AgentState) -> dict:
        if state.get("degraded"):
            return fixed(_degraded_text(state))
        sources = state.get("sources", [])
        titles = list(dict.fromkeys(s["title"] for s in sources if s.get("role") == "retrieved"))
        focus = f'"{titles[0]}"' if len(titles) == 1 else "the work in the sources"
        system, version = prompts.render(
            "critic", state.get("user_id"), reader=_reader(state), focus=focus
        )
        gen = await generate(
            deps,
            system,
            sources,
            _history(state),
            state["query"],
            agent="critic",
            tags={"prompt": version},
            trailer=False,
        )
        return finish("critic", gen.text, sources, gen, version, checks_mod.critic_checks(gen.text))

    return critic


# ================================================================== Synthesizer


def make_synthesizer(deps):
    async def synthesizer(state: AgentState) -> dict:
        if state.get("degraded"):
            return fixed(_degraded_text(state))
        sources = state.get("sources", [])
        system, version = prompts.render(
            "synthesizer", state.get("user_id"), reader=_reader(state), topic=state["query"]
        )
        gen = await generate(
            deps,
            system,
            sources,
            _history(state),
            state["query"],
            agent="synthesizer",
            tags={"prompt": version},
            trailer=False,
        )
        citations, _ = resolve_citations(gen.text, sources)
        papers = {c["paper_id"] for c in citations}
        checks = checks_mod.synthesizer_checks(gen.text, papers)
        return finish("synthesizer", gen.text, sources, gen, version, checks)

    return synthesizer


# ================================================================== Advisor


def make_advisor(deps):
    async def advisor(state: AgentState) -> dict:
        if state.get("degraded") == "retrieval_unavailable":
            return fixed(RETRIEVAL_UNAVAILABLE)
        sources = state.get("sources", [])
        weak = list(state.get("weak_concepts", []))
        profile = state.get("profile") or {}
        interests = set(profile.get("interests") or [])
        candidates: dict[str, advisor_mod.Candidate] = {}
        for rank, s in enumerate(sources):
            if s["paper_id"] in candidates:
                continue
            candidates[s["paper_id"]] = advisor_mod.Candidate(
                paper_id=s["paper_id"],
                title=s["title"],
                snippet=s["content"],
                rank=rank,
                published=s.get("published"),
                url=s["url"],
                concepts=vocab.extract_concepts(f"{s['title']} {s['content']}", limit=8),
            )
        if deps.graph is not None and candidates and settings.GRAPH_ENABLED:
            try:
                async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
                    await _graph_signals(deps.graph, candidates)
            except Exception as e:  # graph signals are a bonus; rank without them
                logger.warning("Advisor graph signals unavailable", error=repr(e))
        picked = advisor_mod.rank_candidates(
            list(candidates.values()), set(weak), interests, settings.ADVISOR_MAX_RECOMMENDATIONS
        )
        text = advisor_mod.render(picked, state.get("due_reviews", []), weak)
        rec_sources = [
            Source(
                n=n,
                chunk_id=f"rec:{c.paper_id}",
                paper_id=c.paper_id,
                title=c.title,
                section="recommendation",
                content=c.snippet,
                score=round(c.score, 4),
                url=c.url or f"https://arxiv.org/abs/{c.paper_id}",
                math_fidelity="latex",
                role="recommendation",
                published=c.published,
            )
            for n, c in enumerate(picked, start=1)
        ]
        result = fixed(text)
        citations, _ = resolve_citations(text, rec_sources)
        return {
            **result,
            "sources": rec_sources,
            "citations": citations,
            "provider": "advisor",
            "model": "deterministic-ranker",
            "prompt_version": "advisor@rules-v1",
            "checks": {"has_recommendations": bool(picked)},
        }

    return advisor


async def _graph_signals(graph, candidates: dict[str, advisor_mod.Candidate]) -> None:
    """Influence for every candidate; prerequisites of the top three become candidates too."""
    top = list(candidates)[:3]
    for paper_id in top:
        for p in await graph.prerequisites(paper_id, 1, 3):
            cand = candidates.get(p["arxiv_id"])
            if cand is None:
                cand = candidates[p["arxiv_id"]] = advisor_mod.Candidate(
                    paper_id=p["arxiv_id"],
                    title=p["title"],
                    snippet=p["title"],
                    rank=len(candidates),
                    published=p.get("published"),
                    concepts=vocab.extract_concepts(p["title"], limit=8),
                )
            cand.prerequisite_of.append(candidates[paper_id].title)
    influence = await graph.influence(list(candidates))
    for pid, rank in influence.items():
        candidates[pid].pagerank = rank


# ================================================================== Tutor


def make_tutor(deps):
    async def tutor(state: AgentState) -> dict:
        action = state.get("tutor_action") or "explain"
        ts = state.get("tutor_state") or {}
        plan = state.get("plan") or {}
        query = state["query"]

        if action == "skip":
            key = (ts.get("pending") or {}).get("answer_key")
            text = "Okay, let's skip that one."
            if key:
                text += f" For reference, the answer I was looking for: {key}"
            text += ' Ask me anything, say "quiz me" for a review, or "next" to move on.'
            new_ts, events, _ = tutor_mod.transition(
                ts, "skip", meta=None, text="", topic=query, concept=None, sources=[]
            )
            return fixed(text, tutor_state=new_ts, learning_events=events, verdict=None)
        if action == "review" and not plan.get("concept"):
            return fixed(NOTHING_DUE, tutor_state=ts or None, learning_events=[])
        if state.get("degraded") == "reading_done":
            ended = {**ts, "reading": None, "pending": None}
            return fixed(READING_DONE, tutor_state=ended, learning_events=[])
        if state.get("degraded"):
            return fixed(_degraded_text(state), tutor_state=ts or None, learning_events=[])

        sources = state.get("sources", [])
        level = mastery_level(state.get("profile"), state.get("mastery"))
        offset = ts.get("level", 0) - (1 if action == "reexplain" else 0)
        profile = state.get("profile") or {}
        variables = {
            "reader": _reader(state),
            "pitch": tutor_mod.pitch(level, offset),
            "max_words": settings.TUTOR_EXPLAIN_MAX_WORDS,
        }
        pending = ts.get("pending") or {}
        reading = state.get("reading") or {}
        equation = state.get("equation") or {}
        concept = None
        topic = query

        if action == "explain":
            template = "tutor_explain"
            concept = next(iter(state.get("concepts") or []), None)
            variables |= {"topic": query, "math": profile.get("math_depth") != "light"}
        elif action == "grade":
            template = "tutor_grade"
            variables |= {
                "question": pending.get("question", ""),
                "answer_key": pending.get("answer_key"),
                "answer": query,
                "hints_left": ts.get("hints_used", 0) < settings.TUTOR_MAX_HINTS,
                "is_last": tutor_mod.is_last_step(ts),
            }
        elif action == "hint":
            template = "tutor_hint"
            variables |= {
                "question": pending.get("question", ""),
                "answer_key": pending.get("answer_key"),
                "hints_used": ts.get("hints_used", 0),
            }
        elif action == "reexplain":
            template = "tutor_reexplain"
            topic = ts.get("topic") or query
            previous = next(
                (
                    m["content"]
                    for m in reversed(state.get("history", []))
                    if m["role"] == "assistant"
                ),
                "",
            )
            variables |= {"topic": topic, "previous": previous[:1200]}
        elif action == "review":
            template = "tutor_review"
            concept = plan["concept"]
            topic = vocab.label(concept)
            variables |= {"topic": topic}
        elif action in ("read", "read_next"):
            template = "tutor_reading"
            topic = reading.get("title", query)
            variables |= {
                "paper_title": reading.get("title", ""),
                "paper_id": reading.get("paper_id", ""),
                "section": reading.get("section", ""),
                "index": reading.get("index", 0),
                "total": reading.get("total", 1),
                "is_first": reading.get("index", 0) == 0,
                "is_last": reading.get("index", 0) + 1 >= reading.get("total", 1),
                "prerequisites": reading.get("prerequisites", []),
            }
        else:  # equation
            template = "tutor_equation"
            topic = (
                f"equation {equation.get('number') or ''} of {equation.get('title', '')}".strip()
            )
            variables |= {
                "paper_title": equation.get("title", ""),
                "paper_id": equation.get("paper_id", ""),
                "section": equation.get("section", ""),
                "latex": equation.get("latex", ""),
                "number": equation.get("number"),
            }

        system, version = prompts.render(template, state.get("user_id"), **variables)
        gen = await generate(
            deps,
            system,
            sources,
            _history(state),
            query,
            agent="tutor",
            tags={"action": action, "prompt": version},
            trailer=True,
        )
        new_ts, events, verdict = tutor_mod.transition(
            ts,
            action,
            meta=gen.meta,
            text=gen.text,
            topic=topic,
            concept=concept,
            sources=sources,
            reading={k: v for k, v in reading.items() if k != "prerequisites"} or None,
            equation=equation or None,
        )
        wrapped_up = action == "grade" and verdict == "correct" and tutor_mod.is_last_step(ts)
        checks = checks_mod.tutor_checks(gen.text, gen.meta, expects_question=not wrapped_up)
        logger.info(
            "Tutor turn", action=action, verdict=verdict, step=new_ts and new_ts.get("step")
        )
        return finish(
            "tutor",
            gen.text,
            sources,
            gen,
            version,
            checks,
            tutor_state=new_ts,
            learning_events=events,
            verdict=verdict,
        )

    return tutor
