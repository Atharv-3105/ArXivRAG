import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agents.prompts import validate_templates
from src.api.errors import install_error_handlers
from src.api.frontend import dist_dir, mount_frontend
from src.api.routes import auth, chat, health, me, papers
from src.api.services import Services, create_services
from src.core.config import settings
from src.core.logging import TraceIDMiddleware, setup_logging
from src.core.metrics import MetricsMiddleware, RuntimeCollector


def create_app(services: Services | None = None) -> FastAPI:
    """`services` lets tests inject fakes; production builds them in the lifespan."""
    setup_logging()
    logger = structlog.get_logger("arxiv-project")
    logger.info("Initializing application", debug=settings.DEBUG)
    # A broken prompt template or a PROMPT_VARIANTS entry without a file stops startup here,
    # instead of failing on some user's turn.
    templates = validate_templates()
    logger.info("Prompt templates valid", templates={k: v for k, v in templates.items()})

    if settings.LANGSMITH_TRACING:
        # Opt-in: traces leave the machine. LangGraph/langsmith read these env vars.
        os.environ.setdefault("LANGSMITH_TRACING", "true")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = services is None
        app.state.services = services or await create_services()
        await app.state.services.start()  # MemoryManager worker
        RuntimeCollector.services = app.state.services  # queue depth, breakers at scrape time
        logger.info("Application ready")
        try:
            yield
        finally:
            if owned:
                await app.state.services.aclose()
            elif app.state.services.memory is not None:
                await app.state.services.memory.stop()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    app.add_middleware(TraceIDMiddleware)
    app.add_middleware(MetricsMiddleware)  # outermost: times everything, including errors
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(health.router)  # root /health, /health/ready, /metrics for platforms
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(me.router, prefix="/api/v1")
    app.include_router(papers.router, prefix="/api/v1")
    app.include_router(papers.ui_router)
    if mount_frontend(app):  # last: its fallback route catches every other path
        logger.info("Serving the web app", dist=str(dist_dir()))
    return app


app = create_app()
