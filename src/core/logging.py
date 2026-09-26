import logging
from uuid import uuid4

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.core.config import settings

TRACE_HEADER = "x-trace-id"


def setup_logging(debug: bool = False) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
            if (debug or settings.DEBUG)
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.LOG_LEVEL)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def current_trace_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("trace_id")


class TraceIDMiddleware:
    """Pure ASGI middleware (not BaseHTTPMiddleware) so it works with streaming/SSE responses.

    Takes the caller's X-Trace-ID or mints one, binds it to every log line of the request, and
    echoes it on the response so a user-reported error can be matched to server logs.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(TRACE_HEADER.encode())
        trace_id = incoming.decode()[:128] if incoming else str(uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id, method=scope.get("method"), path=scope.get("path")
        )

        async def send_with_trace(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"X-Trace-ID", trace_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_trace)
