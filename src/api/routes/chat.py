"""POST /chat (JSON) and POST /chat/stream (Server-Sent Events).

SSE rather than WebSockets: the stream is one-way (server -> client), works through every proxy
and CDN as plain HTTP, reconnects natively in browsers, and needs no connection state on a
free-tier host. POST, because a question does not belong in a URL (logs, history, length limits).

Stream protocol (each event is `event: <name>` + `data: <json>`):
  meta      {conversation_id, trace_id}
  intent    {intent, confidence}
  sources   {sources: [...]}                  the retrieved context, before any token
  provider  {provider, model}
  token     {text}                            repeated
  done      {answer, citations, ungrounded, degraded, provider, model, timings}
  error     {code, message, retry_after?}     terminal; replaces done
"""

import json
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agents.state import Citation, Source
from src.api.errors import retry_after_of
from src.api.services import Services
from src.core.config import settings
from src.core.exceptions import ArxivMindError
from src.core.logging import current_trace_id

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=settings.MAX_MESSAGE_CHARS)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    intent: str
    citations: list[Citation]
    sources: list[Source]
    ungrounded: list[int]
    degraded: str | None
    provider: str
    model: str
    trace_id: str | None


def _services(request: Request) -> Services:
    return request.app.state.services


async def _load_history(services: Services, conversation_id: str) -> list[dict]:
    try:
        return await services.conversations.history(conversation_id, settings.CHAT_HISTORY_MESSAGES)
    except Exception as e:  # history is context, not a requirement
        logger.error("Could not load history", conversation_id=conversation_id, error=str(e))
        return []


async def _save_turn(services: Services, conversation_id: str, query: str, state: dict) -> None:
    trace_id = current_trace_id()
    turn = [
        {"role": "user", "content": query, "trace_id": trace_id},
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
        await services.conversations.append(conversation_id, turn, title=query[:80])
    except Exception as e:
        logger.error("Could not save turn", conversation_id=conversation_id, error=str(e))


def _input(query: str, history: list[dict]) -> dict:
    return {"query": query, "history": history, "timings": {}}


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    services = _services(request)
    conversation_id = str(body.conversation_id or uuid4())
    history = await _load_history(services, conversation_id)
    state = await services.graph.ainvoke(_input(body.message, history))
    await _save_turn(services, conversation_id, body.message, state)
    return ChatResponse(
        conversation_id=conversation_id,
        answer=state.get("response", ""),
        intent=state.get("intent", "chat"),
        citations=state.get("citations", []),
        sources=state.get("sources", []),
        ungrounded=state.get("ungrounded", []),
        degraded=state.get("degraded"),
        provider=state.get("provider", ""),
        model=state.get("model", ""),
        trace_id=current_trace_id(),
    )


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    services = _services(request)
    conversation_id = str(body.conversation_id or uuid4())
    history = await _load_history(services, conversation_id)
    trace_id = current_trace_id()

    async def events() -> AsyncIterator[str]:
        yield sse("meta", {"conversation_id": conversation_id, "trace_id": trace_id})
        final: dict = {}
        try:
            async for mode, payload in services.graph.astream(
                _input(body.message, history), stream_mode=["custom", "updates"]
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

        await _save_turn(services, conversation_id, body.message, final)
        yield sse(
            "done",
            {
                "answer": final.get("response", ""),
                "citations": final.get("citations", []),
                "ungrounded": final.get("ungrounded", []),
                "degraded": final.get("degraded"),
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
