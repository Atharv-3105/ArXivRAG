import logging 
import structlog 
from uuid import uuid4 
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request 
from starlette.responses import Response 
from starlette.types import ASGIApp
from src.core.config import settings 

def setup_logging(debug: bool = False):
    structlog.configure(
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt = "iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer() if not settings.DEBUG else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class = structlog.make_filtering_bound_logger(getattr(logging, settings.LOG_LEVEL)),
        context_class = dict,
        logger_factory = structlog.PrintLoggerFactory(),
        cache_logger_on_first_use = False
    )
    
class TraceIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("X-Trace-ID") or str(uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id, method=request.method, path=request.url.path)
        
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response