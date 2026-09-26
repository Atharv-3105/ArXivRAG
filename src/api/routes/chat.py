"""POST /chat (JSON), POST /chat/stream (Server-Sent Events), POST /feedback.

SSE rather than WebSockets: the stream is one-way (server -> client), works through every proxy
and CDN as plain HTTP, reconnects natively in browsers, and needs no connection state on a
free-tier host. POST, because a question does not belong in a URL (logs, history, length limits).

Stream protocol (each event is `event: <name>` + `data: <json>`):
  meta      {conversation_id, trace_id, concepts}
  intent    {intent, confidence}
  sources   {sources: [...]}                  the retrieved context, before any token
  provider  {provider, model}
  token     {text}                            repeated
  done      {answer, citations, ungrounded, degraded, provider, model, timings}
  error     {code, message, retry_after?}     terminal; replaces done

Per turn (Phase 3): auth -> rate/quota limits -> history (owner-checked) -> profile + mastery of
the query's concepts -> graph -> save turn -> MemoryManager.submit (never awaited).
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agents.state import Citation, Source
from src.api.deps import current_user, services
from src.api.errors import retry_after_of
from src.api.limits import enforce_chat_limits
from src.api.services import Services
from src.auth.users import UserRecord
from src.core.config import settings
from src.core.exceptions import ArxivMindError, NotFoundError
from src.core.logging import current_trace_id
from src.memory import concepts as vocab
from src.memory.manager import MemoryEvent
from src.memory.model import effective_mastery, prior_for

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=settings.MAX_MESSAGE_CHARS)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    intent: str
    concepts: list[str]
    citations: list[Citation]
    sources: list[Source]
    ungrounded: list[int]
    degraded: str | None
    graph_status: str
    provider: str
    model: str
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

    concepts = vocab.extract_concepts(body.message)
    profile, mastery = await _reader_context(svc, user.id, concepts)
    return Turn(
        user=user,
        conversation_id=conversation_id,
        query=body.message,
        concepts=concepts,
        graph_input={
            "query": body.message,
            "history": history,
            "profile": profile,
            "mastery": mastery,
            "concepts": concepts,
            "timings": {},
        },
    )


async def _reader_context(
    svc: Services, user_id: int, concepts: list[str]
) -> tuple[dict, dict[str, float]]:
    """Profile + effective mastery of this query's concepts. Best effort: defaults on failure.
    (Plan: Redis hot cache. Here: two primary-key reads on Postgres, ~ms - ADR 0006.)"""
    try:
        state = await svc.profiles.get_profile(user_id)
        rows = await svc.profiles.get_mastery(user_id, concepts) if concepts else []
    except Exception as e:
        logger.error("Could not load profile; using defaults", user_id=user_id, error=str(e))
        return {}, {}
    now = datetime.now(UTC)
    prior = prior_for(state.profile)
    half_life = settings.MEMORY_DECAY_HALF_LIFE_DAYS
    mastery = {
        s.concept: round(effective_mastery(s, prior, now, half_life), 3)
        for s in rows[: settings.PROFILE_CONCEPTS_IN_PROMPT]
    }
    profile = {k: state.profile.get(k) for k in ("expertise_level", "math_depth", "goals")}
    return profile, mastery


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
        },
    ]
    try:
        await svc.conversations.append(
            turn.user.id, turn.conversation_id, messages, title=turn.query[:80]
        )
    except Exception as e:
        logger.error("Could not save turn", conversation_id=turn.conversation_id, error=str(e))
    if turn.concepts and not state.get("degraded") and svc.memory is not None:
        svc.memory.submit(
            MemoryEvent(
                user_id=turn.user.id,
                kind="chat_turn",
                payload={"intent": state.get("intent", "chat"), "concepts": turn.concepts},
            )
        )


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
        concepts=turn.concepts,
        citations=state.get("citations", []),
        sources=state.get("sources", []),
        ungrounded=state.get("ungrounded", []),
        degraded=state.get("degraded"),
        graph_status=state.get("graph_status", "off"),
        provider=state.get("provider", ""),
        model=state.get("model", ""),
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
        try:
            async for mode, payload in svc.graph.astream(
                turn.graph_input, stream_mode=["custom", "updates"]
            ):
                if mode == "custom":
                    yield sse(payload["type"], {k: v for k, v in payload.items() if k != "type"})
                    continue
                for node, update in payload.items():
                    update = update or {}
                    # "updates" mode yields raw node output: apply the timings reducer ourselves.
                    timings = {**final.get("timings", {}), **update.get("timings", {})}
                    final.update(update)
                    final["timings"] = timings
                    if node == "classify":
                        yield sse(
                            "intent",
                            {"intent": update["intent"], "confidence": update["intent_confidence"]},
                        )
        except ArxivMindError as e:
            logger.error("Stream failed", code=e.code, error=e.message)
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
