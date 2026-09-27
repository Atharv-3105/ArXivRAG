"""POST /chat (JSON), POST /chat/stream (Server-Sent Events), POST /feedback.

SSE rather than WebSockets: the stream is one-way (server -> client), works through every proxy
and CDN as plain HTTP, reconnects natively in browsers, and needs no connection state on a
free-tier host. POST, because a question does not belong in a URL (logs, history, length limits).

Stream protocol (each event is `event: <name>` + `data: <json>`):
  meta      {conversation_id, trace_id, concepts}
  intent    {intent, confidence, agent, action}
  sources   {sources: [...]}                  the retrieved context, before any token
  provider  {provider, model}
  token     {text}                            repeated
  done      {answer, citations, ungrounded, degraded, provider, model, agent, prompt_version,
             checks, verdict, tutor, timings}
  error     {code, message, retry_after?}     terminal; replaces done

Per turn (Phase 3): auth -> rate/quota limits -> history (owner-checked) -> profile + mastery of
the query's concepts -> graph -> save turn -> MemoryManager.submit (never awaited).
"""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agents import tutor as tutor_mod
from src.agents.state import Citation, Source
from src.api.deps import current_user, services
from src.api.errors import retry_after_of
from src.api.limits import enforce_chat_limits
from src.api.services import Services
from src.auth.users import UserRecord
from src.core import metrics
from src.core.config import settings
from src.core.exceptions import ArxivMindError, NotFoundError
from src.core.logging import current_trace_id
from src.memory import concepts as vocab
from src.memory.manager import MemoryEvent
from src.memory.model import effective_mastery, prior_for

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Chat"])


Action = Literal["answer", "hint", "dont_understand", "skip", "review", "read", "next", "equation"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=settings.MAX_MESSAGE_CHARS)
    conversation_id: UUID | None = None
    # Explicit tutor actions for UI buttons ("Hint", "I don't understand", "Next section", ...).
    # Free text works too; an explicit action just removes the guessing.
    action: Action | None = None
    paper_id: str | None = Field(default=None, max_length=40, description="for read / equation")
    equation: str | None = Field(default=None, max_length=10, description='e.g. "3"')


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    intent: str
    agent: str
    action: str | None
    concepts: list[str]
    citations: list[Citation]
    sources: list[Source]
    ungrounded: list[int]
    degraded: str | None
    graph_status: str
    provider: str
    model: str
    prompt_version: str | None
    checks: dict[str, bool]
    verdict: str | None
    tutor: dict | None
    guardrail: dict | None
    trace_id: str | None


class Feedback(BaseModel):
    """`understood`/`confused` are evidence about the learner (they move mastery);
    `helpful`/`not_helpful` rate the answer (recorded for evaluation, mastery untouched)."""

    signal: Literal["understood", "confused", "helpful", "not_helpful"]
    concepts: list[str] = Field(default_factory=list, max_length=10)
    conversation_id: UUID | None = None


@dataclass
class Turn:
    user: UserRecord
    conversation_id: str
    query: str
    concepts: list[str]
    graph_input: dict


async def _prepare(svc: Services, user: UserRecord, body: ChatRequest) -> Turn:
    await enforce_chat_limits(user.id, user.is_admin, svc.limiter, svc.conversations)
    conversation_id = str(body.conversation_id or uuid4())
    try:
        history = await svc.conversations.history(
            user.id, conversation_id, settings.CHAT_HISTORY_MESSAGES
        )
    except NotFoundError:
        raise  # someone else's conversation: never degrade into answering
    except Exception as e:  # history is context, not a requirement
        logger.error("Could not load history", conversation_id=conversation_id, error=str(e))
        history = []

    try:
        tutor_state = await svc.conversations.get_state(user.id, conversation_id)
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("Could not load tutor state", conversation_id=conversation_id, error=str(e))
        tutor_state = None

    concepts = vocab.extract_concepts(body.message)
    reader = await _reader_context(svc, user.id, concepts)
    args = {k: v for k, v in (("paper_id", body.paper_id), ("equation", body.equation)) if v}
    return Turn(
        user=user,
        conversation_id=conversation_id,
        query=body.message,
        concepts=concepts,
        graph_input={
            "query": body.message,
            "user_id": user.id,
            "history": history,
            "concepts": concepts,
            "tutor_state": tutor_state,
            "action": body.action,
            "action_args": args,
            "timings": {},
            **reader,
        },
    )


async def _reader_context(svc: Services, user_id: int, concepts: list[str]) -> dict:
    """Profile, effective mastery (this query's concepts), weak concepts and due reviews.
    Best effort: defaults on failure. (Plan: Redis hot cache. Here: three indexed reads on
    Postgres - ADR 0006.)"""
    try:
        state = await svc.profiles.get_profile(user_id)
        rows = await svc.profiles.get_mastery(user_id)
        now = datetime.now(UTC)
        due = await svc.profiles.due_reviews(user_id, now, 5)
    except Exception as e:
        logger.error("Could not load profile; using defaults", user_id=user_id, error=str(e))
        return {"profile": {}, "mastery": {}, "weak_concepts": [], "due_reviews": []}
    prior = prior_for(state.profile)
    half_life = settings.MEMORY_DECAY_HALF_LIFE_DAYS
    effective = {s.concept: round(effective_mastery(s, prior, now, half_life), 3) for s in rows}
    mastery = {c: effective[c] for c in concepts if c in effective}
    mastery = dict(list(mastery.items())[: settings.PROFILE_CONCEPTS_IN_PROMPT])
    weak = sorted(
        (c for c, m in effective.items() if m < settings.MASTERY_WEAK_BELOW),
        key=lambda c: effective[c],
    )
    inferred = sorted(
        (state.profile.get("inferred_interests") or {}).items(), key=lambda kv: -kv[1]
    )
    interests = list(
        dict.fromkeys([*(state.profile.get("interests") or []), *(c for c, _ in inferred[:5])])
    )
    profile = {k: state.profile.get(k) for k in ("expertise_level", "math_depth", "goals")}
    return {
        "profile": {**profile, "interests": interests},
        "mastery": mastery,
        "weak_concepts": weak,
        "due_reviews": [d.concept for d in due],
    }


async def _finish(svc: Services, turn: Turn, state: dict) -> None:
    """Save the turn, then hand the learning signal to the MemoryManager (not awaited)."""
    trace_id = current_trace_id()
    messages = [
        {"role": "user", "content": turn.query, "trace_id": trace_id},
        {
            "role": "assistant",
            "content": state.get("response", ""),
            "intent": state.get("intent"),
            "citations": state.get("citations", []),
            "provider": state.get("provider"),
            "trace_id": trace_id,
            "agent": state.get("agent"),
            "prompt_version": state.get("prompt_version"),
            "checks": {
                **(state.get("checks") or {}),
                "guardrail_approved": (state.get("guardrail") or {}).get("approved", True),
            },
        },
    ]
    _record_metrics(state)
    try:
        await svc.conversations.append(
            turn.user.id,
            turn.conversation_id,
            messages,
            title=turn.query[:80],
            state=state.get("tutor_state"),
        )
    except Exception as e:
        logger.error("Could not save turn", conversation_id=turn.conversation_id, error=str(e))
    if svc.memory is None or state.get("degraded"):
        return
    # A fresh question is (weak) evidence about its concepts; tutor follow-ups (answers, hints,
    # "I don't understand") carry their own, stronger signals from the tutor state machine.
    if turn.concepts and state.get("tutor_action") in (None, "explain"):
        svc.memory.submit(
            MemoryEvent(
                user_id=turn.user.id,
                kind="chat_turn",
                payload={"intent": state.get("intent", "chat"), "concepts": turn.concepts},
            )
        )
    for event in state.get("learning_events") or []:
        svc.memory.submit(MemoryEvent(user_id=turn.user.id, **event))


def _record_metrics(state: dict) -> None:
    agent = state.get("agent") or "general"
    outcome = "degraded" if state.get("degraded") else "ok"
    metrics.CHAT_TURNS.labels(
        agent, state.get("tutor_action") or "-", state.get("provider") or "-", outcome
    ).inc()
    for check, ok in (state.get("checks") or {}).items():
        if not ok:
            metrics.CHECK_FAILURES.labels(agent, check).inc()
    if state.get("graph_status"):
        metrics.GRAPH_RETRIEVAL.labels(state["graph_status"]).inc()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
) -> ChatResponse:
    turn = await _prepare(svc, user, body)
    state = await svc.graph.ainvoke(turn.graph_input)
    await _finish(svc, turn, state)
    return ChatResponse(
        conversation_id=UUID(turn.conversation_id),
        answer=state.get("response", ""),
        intent=state.get("intent", "chat"),
        agent=state.get("agent", "general"),
        action=state.get("tutor_action"),
        concepts=turn.concepts,
        citations=state.get("citations", []),
        sources=state.get("sources", []),
        ungrounded=state.get("ungrounded", []),
        degraded=state.get("degraded"),
        graph_status=state.get("graph_status", "off"),
        provider=state.get("provider", ""),
        model=state.get("model", ""),
        prompt_version=state.get("prompt_version"),
        checks=state.get("checks") or {},
        verdict=state.get("verdict"),
        tutor=tutor_mod.public_view(state.get("tutor_state")),
        guardrail=state.get("guardrail"),
        trace_id=current_trace_id(),
    )


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
) -> StreamingResponse:
    # Limits, auth and ownership fail as normal HTTP errors, before the stream starts.
    started = time.perf_counter()
    turn = await _prepare(svc, user, body)
    trace_id = current_trace_id()

    async def events() -> AsyncIterator[str]:
        yield sse(
            "meta",
            {
                "conversation_id": turn.conversation_id,
                "trace_id": trace_id,
                "concepts": turn.concepts,
            },
        )
        final: dict = {}
        first_token_seen = False
        try:
            async for mode, payload in svc.graph.astream(
                turn.graph_input, stream_mode=["custom", "updates"]
            ):
                if mode == "custom":
                    if payload["type"] == "token" and not first_token_seen:
                        first_token_seen = True
                        metrics.FIRST_TOKEN.labels(final.get("agent") or "-").observe(
                            time.perf_counter() - started
                        )
                    yield sse(payload["type"], {k: v for k, v in payload.items() if k != "type"})
                    continue
                for node, update in payload.items():
                    update = update or {}
                    # "updates" mode yields raw node output: apply the timings reducer ourselves.
                    timings = {**final.get("timings", {}), **update.get("timings", {})}
                    final.update(update)
                    final["timings"] = timings
                    if node == "route":
                        yield sse(
                            "intent",
                            {
                                "intent": update["intent"],
                                "confidence": update["intent_confidence"],
                                "agent": update["agent"],
                                "action": update["tutor_action"],
                            },
                        )
        except ArxivMindError as e:
            logger.error("Stream failed", code=e.code, error=e.message)
            metrics.CHAT_TURNS.labels(final.get("agent") or "-", "-", "-", "error").inc()
            data: dict[str, object] = {"code": e.code, "message": e.message}
            if (retry_after := retry_after_of(e)) is not None:
                data["retry_after"] = retry_after
            yield sse("error", data)
            return
        except Exception:
            logger.exception("Stream failed unexpectedly")
            yield sse("error", {"code": "INTERNAL_ERROR", "message": "Something went wrong."})
            return

        await _finish(svc, turn, final)
        yield sse(
            "done",
            {
                "answer": final.get("response", ""),
                "citations": final.get("citations", []),
                "ungrounded": final.get("ungrounded", []),
                "degraded": final.get("degraded"),
                "graph_status": final.get("graph_status", "off"),
                "provider": final.get("provider", ""),
                "model": final.get("model", ""),
                "agent": final.get("agent"),
                "prompt_version": final.get("prompt_version"),
                "checks": final.get("checks") or {},
                "verdict": final.get("verdict"),
                "tutor": tutor_mod.public_view(final.get("tutor_state")),
                "guardrail": final.get("guardrail"),
                "timings": final.get("timings", {}),
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # no-transform/X-Accel-Buffering: stop proxies (nginx, Render, Cloudflare) from buffering
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", status_code=status.HTTP_202_ACCEPTED)
async def feedback(
    body: Feedback,
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
) -> dict:
    """202: accepted for asynchronous processing - the profile changes shortly after."""
    concepts = [c for c in body.concepts if vocab.is_known(c)]
    queued = svc.memory is not None and svc.memory.submit(
        MemoryEvent(
            user_id=user.id,
            kind="feedback",
            payload={
                "signal": body.signal,
                "concepts": concepts,
                "conversation_id": str(body.conversation_id) if body.conversation_id else None,
            },
        )
    )
    return {"queued": bool(queued), "concepts": concepts}
