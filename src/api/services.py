"""Long-lived objects the API shares across requests, built once in the app lifespan.

Tests build a `Services` with fakes and hand it to `create_app(services=...)`, so no test needs
Qdrant, Postgres or an API key.
"""

from dataclasses import dataclass
from typing import Any

import structlog

from src.agents.graph import ChatDeps, Retriever, build_graph
from src.core.exceptions import IndexError
from src.llm.registry import build_router
from src.llm.router import LLMRouter
from src.storage.conversations import ConversationStore, PgConversationStore
from src.storage.vector_db import VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class Services:
    llm: LLMRouter
    retriever: Retriever  # VectorStore in production
    conversations: ConversationStore
    graph: Any = None  # compiled LangGraph

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = build_graph(ChatDeps(llm=self.llm, retriever=self.retriever))

    async def aclose(self) -> None:
        await self.llm.aclose()
        close = getattr(self.retriever, "close", None)
        if close is not None:
            await close()


async def create_services() -> Services:
    """Dependencies that are down at startup degrade the API instead of stopping it:
    a free-tier database that is still waking up must not crash-loop the web process.
    A *misconfigured* dependency (wrong collection layout) does stop it: that never heals."""
    from src.storage.database import init_db_schema

    try:
        await init_db_schema()
    except Exception as e:
        logger.error("Database unavailable at startup; chat history is best effort", error=str(e))

    store = VectorStore()
    try:
        await store.initialize()
    except IndexError:
        raise
    except Exception as e:
        logger.error("Qdrant unavailable at startup; answers will degrade", error=str(e))

    return Services(llm=build_router(), retriever=store, conversations=PgConversationStore())
