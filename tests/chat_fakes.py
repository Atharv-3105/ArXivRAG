"""Offline stand-ins for the chat loop: a real LLMRouter over fake adapters, a scripted retriever."""

from src.api.services import Services
from src.core.exceptions import RetrievalError
from src.llm.base import ErrorKind, LLMRequest, ProviderError
from src.llm.mock import MockAdapter
from src.llm.router import LLMRouter, ProviderLimits
from src.storage.conversations import InMemoryConversationStore
from src.storage.vector_db import SearchHit


def hit(paper_id: str, n: int, content: str | None = None, **meta) -> SearchHit:
    return SearchHit(
        chunk_id=f"{paper_id}-{n}",
        paper_id=paper_id,
        title=f"Title {paper_id}",
        section="3 Method",
        content=content or f"LoRA injects trainable low-rank matrices ({paper_id} chunk {n}).",
        score=1.0 / (n + 1),
        metadata={"math_fidelity": "latex", "parse_method": "html", "anchor": f"S3.p{n}", **meta},
    )


class FakeRetriever:
    def __init__(self, hits: list[SearchHit] | None = None, error: Exception | None = None):
        self.hits = hits or []
        self.error = error
        self.queries: list[str] = []

    async def search(self, query, top_k=None, paper_ids=None):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.hits[: top_k or len(self.hits)]


class RecordingAdapter(MockAdapter):
    """The deterministic mock, plus a record of what it was asked."""

    def __init__(self):
        super().__init__()
        self.requests: list[LLMRequest] = []

    async def complete(self, request):
        self.requests.append(request)
        return await super().complete(request)

    def stream(self, request):
        self.requests.append(request)
        return super().stream(request)


class DeadAdapter(MockAdapter):
    """Every call fails with a bad key: the router opens the breaker and has nothing left."""

    def __init__(self):
        super().__init__(name="dead")

    async def complete(self, request):
        raise ProviderError(self.name, ErrorKind.AUTH, "bad key", status_code=401)

    async def stream(self, request):
        raise ProviderError(self.name, ErrorKind.AUTH, "bad key", status_code=401)
        yield  # pragma: no cover - makes this an async generator


def router(adapter=None) -> LLMRouter:
    return LLMRouter([(adapter or RecordingAdapter(), ProviderLimits())])


def make_services(hits=None, retrieval_error=None, adapter=None) -> Services:
    return Services(
        llm=router(adapter),
        retriever=FakeRetriever(hits, retrieval_error),
        conversations=InMemoryConversationStore(),
    )


RETRIEVAL_DOWN = RetrievalError("qdrant down", code="VECTOR_SEARCH_ERROR")
