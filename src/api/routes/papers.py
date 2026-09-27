"""Paper and citation-graph endpoints (public, read-only).

Public because the data is public arXiv metadata and no LLM is involved; still rate-limited per
IP and size-capped, because a free-tier database is the scarce resource. Old-style arXiv ids
contain a slash ("hep-th/9901001"), hence the `:path` converters; the fixed routes (/top,
/missing) are declared before the catch-all.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from src.api.deps import services
from src.api.services import Services
from src.core import metrics
from src.core.config import settings
from src.core.exceptions import NotFoundError, RateLimitError, RetrievalError

router = APIRouter(prefix="/papers", tags=["Papers"])
ui_router = APIRouter(tags=["UI"])

_VIEWER = Path(__file__).resolve().parent.parent / "static" / "graph.html"
FUNCTIONS = ("critical", "comparative", "builds_on", "neutral")


def _graph(svc: Services):
    if svc.graph_store is None:
        raise RetrievalError("Citation graph is not configured", code="GRAPH_UNAVAILABLE")
    return svc.graph_store


async def public_limit(request: Request, svc: Services = Depends(services)) -> None:
    # Behind a proxy, run uvicorn with --proxy-headers so client.host is the real client.
    ip = request.client.host if request.client else "unknown"
    wait = svc.public_limiter.hit(ip)
    if wait is not None:
        metrics.RATE_LIMITED.labels("public_ip").inc()
        raise RateLimitError("Too many requests", retry_after=max(1, int(wait) + 1))


async def _paper_or_404(svc: Services, arxiv_id: str) -> dict:
    paper = await _graph(svc).paper(arxiv_id)
    if paper is None:
        raise NotFoundError(f"Paper {arxiv_id} is not in the corpus", code="PAPER_NOT_FOUND")
    return paper


@router.get("/top", dependencies=[Depends(public_limit)])
async def top(limit: int = Query(20, ge=1, le=100), svc: Services = Depends(services)):
    """Most influential papers in the corpus (PageRank; 1.0 = average)."""
    return await _graph(svc).top_influential(limit)


@router.get("/map", dependencies=[Depends(public_limit)])
async def paper_map(limit: int = Query(80, ge=5, le=300), svc: Services = Depends(services)):
    """The most influential papers and the citation edges among them, for the map view."""
    return await _graph(svc).map(limit)


@router.get("/missing", dependencies=[Depends(public_limit)])
async def missing(limit: int = Query(20, ge=1, le=100), svc: Services = Depends(services)):
    """arXiv papers our corpus cites most but does not contain: what to ingest next."""
    return await _graph(svc).missing(limit)


@router.get("/{arxiv_id:path}/graph", dependencies=[Depends(public_limit)])
async def neighborhood(
    arxiv_id: str,
    depth: int = Query(1, ge=1, le=2),
    max_nodes: int = Query(80, ge=5, le=settings.GRAPH_MAX_NODES),
    svc: Services = Depends(services),
):
    """Nodes and edges around a paper, for a force-directed view. Edges carry the strongest
    citation function found ("critical" > "comparative" > "builds_on" > "neutral")."""
    await _paper_or_404(svc, arxiv_id)
    return await _graph(svc).neighborhood(arxiv_id, depth, max_nodes)


@router.get("/{arxiv_id:path}/prerequisites", dependencies=[Depends(public_limit)])
async def prerequisites(
    arxiv_id: str,
    depth: int = Query(settings.GRAPH_MAX_DEPTH, ge=1, le=4),
    limit: int = Query(10, ge=1, le=50),
    svc: Services = Depends(services),
):
    """In-corpus papers this one builds on, nearest first, then most influential."""
    paper = await _paper_or_404(svc, arxiv_id)
    return await _graph(svc).prerequisites(paper["arxiv_id"], depth, limit)


@router.get("/{arxiv_id:path}/citations", dependencies=[Depends(public_limit)])
async def citations(
    arxiv_id: str,
    function: list[str] | None = Query(None, description="critical|comparative|builds_on|neutral"),
    limit: int = Query(20, ge=1, le=100),
    svc: Services = Depends(services),
):
    """What other papers in the corpus say about this one (the citing sentences)."""
    functions = [f for f in function or [] if f in FUNCTIONS] or None
    await _paper_or_404(svc, arxiv_id)
    return await _graph(svc).citing_contexts(arxiv_id, functions, limit)


@router.get("/{arxiv_id:path}", dependencies=[Depends(public_limit)])
async def paper(arxiv_id: str, svc: Services = Depends(services)):
    return await _paper_or_404(svc, arxiv_id)


@ui_router.get("/ui/graph", response_class=HTMLResponse, include_in_schema=False)
async def graph_viewer() -> HTMLResponse:
    return HTMLResponse(_VIEWER.read_text(encoding="utf-8"))
