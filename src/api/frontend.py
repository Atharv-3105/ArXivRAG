"""Serve the built web app (frontend/dist) from the API process (ADR 0014).

One origin for the SPA and the API: no CORS, no second free-tier service, and the API's own
security headers cover the pages. Hashed assets (/assets/*) are immutable and cached for a year;
index.html is never cached, so a deploy is picked up on the next navigation. Unknown non-API paths
return index.html, so client-side routes (/chat/<id>, /map) survive a reload.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.core.config import settings
from src.core.exceptions import NotFoundError

REPO_ROOT = Path(__file__).resolve().parents[2]
_API_PREFIXES = ("api/", "ui/", "health", "metrics", "docs", "redoc", "openapi.json")

# Applied to the app's HTML: the SPA loads only its own scripts, styles and fonts.
# 'unsafe-inline' for styles: KaTeX and the map position elements with inline style attributes.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "font-src 'self' data:; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


class _ImmutableStatic(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def dist_dir() -> Path:
    configured = Path(settings.FRONTEND_DIST)
    return configured if configured.is_absolute() else REPO_ROOT / configured


def mount_frontend(app: FastAPI) -> bool:
    """Register the SPA routes. Call last: the fallback route matches everything else."""
    dist = dist_dir()
    index = dist / "index.html"
    if not index.is_file():
        return False  # API-only (tests, or the frontend was not built)
    if (dist / "assets").is_dir():
        app.mount("/assets", _ImmutableStatic(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        if full_path.startswith(_API_PREFIXES):
            raise NotFoundError("Not found", code="NOT_FOUND")
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(candidate)  # favicon, robots.txt, ...
        return FileResponse(index, headers={"Cache-Control": "no-cache", **SECURITY_HEADERS})

    return True
