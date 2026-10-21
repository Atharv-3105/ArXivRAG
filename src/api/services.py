"""Long-lived objects the API shares across requests, built once in the app lifespan.

Tests build a `Services` with fakes and hand it to `create_app(services=...)`, so no test needs
Qdrant, Postgres or an API key.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.agents.graph import ChatDeps, Retriever, build_graph
from src.api.limits import AuthThrottle, SlidingWindowLimiter
from src.auth.seed import ensure_admin
from src.auth.users import PgUserStore, UserStore
from src.core.config import settings
from src.core.exceptions import IndexError
from src.core.metrics import observe_llm
from src.graph.store import GraphStore, PgGraphStore
from src.llm.registry import build_router
from src.llm.router import LLMRouter
from src.memory.manager import MemoryManager
from src.memory.store import PgProfileStore, ProfileStore
from src.storage.conversations import ConversationStore, PgConversationStore
from src.storage.paper_content import PaperContent, PgPaperContent
from src.storage.vector_db import VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class Services:
    llm: LLMRouter
    retriever: Retriever  # VectorStore in production
    conversations: ConversationStore
    users: UserStore
    profiles: ProfileStore
    graph_store: GraphStore | None = None
    content: PaperContent | None = None
    memory: MemoryManager | None = None
    limiter: SlidingWindowLimiter = field(
        default_factory=lambda: SlidingWindowLimiter(settings.RATE_LIMIT_PER_MINUTE)
    )
    public_limiter: SlidingWindowLimiter = field(
        default_factory=lambda: SlidingWindowLimiter(settings.RATE_LIMIT_PUBLIC_PER_MINUTE)
    )
    auth_throttle: AuthThrottle = field(default_factory=AuthThrottle)
    graph: Any = None  # compiled LangGraph
    # Readiness probes: name -> async callable that raises when the dependency is down.
    probes: dict[str, Callable[[], Awaitable[Any]]] = field(default_factory=dict)
    required_probes: tuple[str, ...] = ("database",)

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = build_graph(
                ChatDeps(
                    llm=self.llm,
                    retriever=self.retriever,
                    graph=self.graph_store,
                    content=self.content,
                )
            )
        if self.memory is None:
            self.memory = MemoryManager(self.profiles)

    async def start(self) -> None:
        assert self.memory is not None
        self.memory.start()  # needs the running event loop, hence not in __post_init__

    async def aclose(self) -> None:
        if self.memory is not None:
            await self.memory.stop()
        await self.llm.aclose()
        close = getattr(self.retriever, "close", None)
        if close is not None:
            await close()


async def create_services() -> Services:
    """Dependencies that are down at startup degrade the API instead of stopping it:
    a free-tier database that is still waking up must not crash-loop the web process.
    A *misconfigured* dependency (wrong collection layout) does stop it: that never heals."""
    from src.storage.database import init_db_schema

    users = PgUserStore()
    try:
        await init_db_schema()
        await ensure_admin(users)
    except Exception as e:
        logger.error("Database unavailable at startup; chat history is best effort", error=str(e))

    store = VectorStore()
    try:
        await store.initialize()
    except IndexError:
        raise
    except Exception as e:
        logger.error("Qdrant unavailable at startup; answers will degrade", error=str(e))

    async def database() -> None:
        from sqlalchemy import text

        from src.storage.database import get_async_session

        async with get_async_session() as session:
            await session.execute(text("SELECT 1"))

    async def qdrant() -> None:
        if not await store.client.collection_exists(store.collection):
            raise RuntimeError("collection missing")

    return Services(
        llm=build_router(observer=observe_llm),
        retriever=store,
        conversations=PgConversationStore(),
        users=users,
        profiles=PgProfileStore(),
        graph_store=PgGraphStore(),
        content=PgPaperContent(),
        probes={"database": database, "qdrant": qdrant},
    )
