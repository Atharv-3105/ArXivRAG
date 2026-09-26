"""One error shape for every failure: {"error": {"code", "message", "trace_id"}}.

The trace id is the same one on the X-Trace-ID header and in every log line of the request, so a
user can report it and an operator can find the exact request in the logs.
"""

import math

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.core.exceptions import ArxivMindError
from src.core.logging import current_trace_id

logger = structlog.get_logger(__name__)


def error_body(code: str, message: str, **extra) -> dict:
    return {"error": {"code": code, "message": message, "trace_id": current_trace_id(), **extra}}


def retry_after_of(exc: Exception) -> int | None:
    value = getattr(exc, "retry_after", None)
    return max(1, math.ceil(value)) if value is not None else None


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ArxivMindError)
    async def domain_error(request: Request, exc: ArxivMindError) -> JSONResponse:
        headers = {}
        retry_after = retry_after_of(exc)
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        if exc.status_code == 401:
            headers["WWW-Authenticate"] = "Bearer"  # RFC 6750
        level = logger.error if exc.status_code >= 500 else logger.info
        level("Request failed", code=exc.code, status=exc.status_code, error=exc.message)
        return JSONResponse(
            error_body(exc.code, exc.message), status_code=exc.status_code, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Plan §4: 400 on validation (FastAPI's default is 422).
        details = [
            {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse(
            error_body("VALIDATION_ERROR", "Invalid request", details=details), status_code=400
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(
            error_body(
                "INTERNAL_ERROR", "Something went wrong. Quote the trace id if you report it."
            ),
            status_code=500,
        )
