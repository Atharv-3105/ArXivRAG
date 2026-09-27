"""Health and metrics endpoints.

/health/live   the process is up (restart it if this fails)
/health/ready  dependencies answer within DEPENDENCY_TIMEOUT_SECONDS (send traffic only if 200).
               The database is required; Qdrant is reported but not required, because the
               product degrades without it (fixed "search unavailable" answers) rather than
               failing - taking the instance out of rotation would turn a partial outage into a
               total one.
/metrics       Prometheus exposition; optional bearer token (METRICS_TOKEN) for public deploys
"""

import asyncio
import hmac
from datetime import UTC, datetime

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.core.config import settings
from src.core.exceptions import AuthError, NotFoundError
from src.core.metrics import REGISTRY

router = APIRouter(tags=["System"])


@router.get("/health")
@router.get("/health/live")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def _probe(check) -> dict:
    try:
        await asyncio.wait_for(check(), settings.DEPENDENCY_TIMEOUT_SECONDS)
        return {"status": "ok"}
    except TimeoutError:
        return {"status": "timeout"}
    except Exception as e:  # report the kind of failure, never connection strings
        return {"status": "error", "error": type(e).__name__}


@router.get("/health/ready")
async def readiness(request: Request):
    services = getattr(request.app.state, "services", None)
    probes = getattr(services, "probes", {}) or {}
    names = list(probes)
    results = await asyncio.gather(*(_probe(probes[n]) for n in names))
    checks = dict(zip(names, results, strict=True))
    required = [n for n in names if n in getattr(services, "required_probes", ())]
    ready = all(checks[n]["status"] == "ok" for n in required)
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "checks": checks},
        status_code=200 if ready else 503,
    )


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    if not settings.METRICS_ENABLED:
        raise NotFoundError("Not found", code="NOT_FOUND")
    if settings.METRICS_TOKEN:
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied, settings.METRICS_TOKEN):  # constant-time compare
            raise AuthError("Metrics token required", code="METRICS_AUTH")
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
