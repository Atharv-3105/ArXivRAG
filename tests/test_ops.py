"""Metrics, health probes, JWT key rotation, fetch-by-id, search modes, eval scoring, chaos."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.config import settings
from src.eval import offline, scoring
from src.llm.base import LLMResult
from src.llm.mock import MockAdapter
from src.llm.router import LLMRouter, ProviderLimits
from tests.chat_fakes import RETRIEVAL_DOWN, FakeRetriever, hit, make_services, signup

# ---------------------------------------------------------------- metrics & health


@pytest.fixture
def api():
    services = make_services([hit("2106.09685v2", 0)])
    with TestClient(create_app(services=services)) as client:
        client.headers.update(signup(client))
        yield client


def test_metrics_exposes_request_and_chat_series(api):
    api.post("/api/v1/chat", json={"message": "Explain how LoRA works"})
    body = api.get("/metrics").text
    assert 'arxivmind_http_requests_total{method="POST",route="/api/v1/chat",status="200"}' in body
    assert "arxivmind_chat_turns_total" in body and 'agent="tutor"' in body
    assert "arxivmind_guardrail_decisions_total" in body
    assert "arxivmind_memory_queue_depth" in body  # runtime collector
    assert "arxivmind_llm_breaker_open" in body


def test_metrics_label_by_route_template_not_raw_path(api):
    api.get("/api/v1/conversations/00000000-0000-0000-0000-000000000000")
    body = api.get("/metrics").text
    assert 'route="/api/v1/conversations/{conversation_id}"' in body
    assert "00000000-0000" not in body


def test_metrics_token(api, monkeypatch):
    monkeypatch.setattr(settings, "METRICS_TOKEN", "s3cret")
    api.headers.pop("Authorization")
    assert api.get("/metrics").status_code == 401
    assert api.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_readiness_reports_probes():
    async def ok():
        return None

    async def down():
        raise ConnectionError("refused")

    services = make_services()
    services.probes = {"database": ok, "qdrant": down}
    with TestClient(create_app(services=services)) as client:
        r = client.get("/health/ready")
        assert r.status_code == 200  # qdrant is not required: the product degrades instead
        assert r.json()["checks"]["qdrant"] == {"status": "error", "error": "ConnectionError"}
        services.probes["database"] = down
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 200


# ---------------------------------------------------------------- JWT rotation


def _token(key: str) -> str:
    now = datetime.now(UTC)
    claims = {"sub": "1", "iat": now, "exp": now + timedelta(hours=1), "iss": settings.APP_NAME}
    return jwt.encode(claims, key, algorithm="HS256")


def test_previous_key_still_verifies_during_rotation(api, monkeypatch):
    old = settings.SECRET_KEY
    old_token = _token(old)
    monkeypatch.setattr(settings, "SECRET_KEY", "a-brand-new-secret-key-0123456789")
    api.headers.pop("Authorization")
    auth = {"Authorization": f"Bearer {old_token}"}
    assert api.get("/api/v1/me", headers=auth).status_code == 401  # rotation without a window
    monkeypatch.setattr(settings, "SECRET_KEY_PREVIOUS", old)
    assert api.get("/api/v1/me", headers=auth).status_code == 200


# ---------------------------------------------------------------- fetch by id, search modes


async def test_fetch_by_id_uses_arxiv_id_list(monkeypatch):
    from src.pipeline import fetcher as fetcher_mod

    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(dict(request.url.params))
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>')

    monkeypatch.setattr(settings, "ARXIV_REQUEST_DELAY_SECONDS", 0)
    f = fetcher_mod.ArxivFetcher()
    f._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await f.fetch_papers(None, id_list=["2106.09685", "1706.03762"])
    await f.aclose()
    assert sent["id_list"] == "2106.09685,1706.03762" and "search_query" not in sent


async def test_single_branch_search_modes(monkeypatch):
    import numpy as np

    from src.storage import vector_db
    from src.storage.embedder import SparseVector

    client = MagicMock()
    client.query_points = AsyncMock(return_value=MagicMock(points=[]))
    monkeypatch.setattr(vector_db, "AsyncQdrantClient", lambda **_: client)
    embedder = MagicMock()
    embedder.encode_query.return_value = np.ones(384)
    embedder.encode_sparse_query.return_value = SparseVector([1], [1.0])
    monkeypatch.setattr(vector_db, "EmbeddingService", lambda: embedder)
    store = vector_db.VectorStore()
    for mode, using in (("dense", "dense"), ("sparse", "bm25")):
        await store.search("q", mode=mode)
        kwargs = client.query_points.await_args.kwargs
        assert kwargs["using"] == using and "prefetch" not in kwargs


# ---------------------------------------------------------------- eval scoring & the gate


def test_ranking_metrics():
    ranked = ["a", "b", "c", "a"]
    assert scoring.recall_at_k(ranked, {"a", "c"}, 2) == 0.5
    assert scoring.mrr(ranked, {"b"}) == 0.5
    assert scoring.ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert scoring.ndcg_at_k(["b", "a"], {"a"}, 2) < 1.0
    assert scoring.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert scoring.macro_f1([("x", "x"), ("y", "x")]) == pytest.approx((0.667 + 0) / 2, abs=0.01)


def test_offline_evaluation_gate():
    """The same gate CI runs: every offline suite meets its threshold."""
    report = offline.run()
    assert report["violations"] == [], report["violations"]


# ---------------------------------------------------------------- chaos: faults never become 500s


class SlowAdapter(MockAdapter):
    def __init__(self):
        super().__init__(name="slow")

    async def complete(self, request):
        await asyncio.sleep(5)
        return LLMResult("late", self.name, self.model)

    async def stream(self, request):
        await asyncio.sleep(5)
        yield "late"


def _broken(*args, **kwargs):
    raise ConnectionError("dependency down")


async def _abroken(*args, **kwargs):
    raise ConnectionError("dependency down")


FAULTS = {
    "qdrant_down": lambda s: setattr(s.retriever, "error", RETRIEVAL_DOWN),
    "history_down": lambda s: (
        setattr(s.conversations, "history", _abroken),
        setattr(s.conversations, "append", _abroken),
        setattr(s.conversations, "get_state", _abroken),
        setattr(s.conversations, "count_user_messages_since", _abroken),
    ),
    "profiles_down": lambda s: (
        setattr(s.profiles, "get_profile", _abroken),
        setattr(s.profiles, "get_mastery", _abroken),
        setattr(s.profiles, "apply_event", _abroken),
    ),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_dependency_faults_degrade_instead_of_failing(fault):
    services = make_services([hit("2106.09685v2", 0)])
    with TestClient(create_app(services=services)) as client:
        client.headers.update(signup(client))
        FAULTS[fault](services)
        r = client.post("/api/v1/chat", json={"message": "Explain how LoRA works"})
        stream = client.post("/api/v1/chat/stream", json={"message": "Explain how LoRA works"})
    assert r.status_code == 200 and r.json()["answer"]
    assert stream.status_code == 200 and "event: done" in stream.text


def test_llm_timeout_fails_over_to_the_next_provider():
    services = make_services([hit("2106.09685v2", 0)])
    services.llm = LLMRouter(
        [(SlowAdapter(), ProviderLimits()), (MockAdapter(), ProviderLimits())],
        call_timeout=0.2,
        first_token_timeout=0.2,
    )
    services.graph = None
    services.__post_init__()  # rebuild the graph around the new router
    with TestClient(create_app(services=services)) as client:
        client.headers.update(signup(client))
        r = client.post("/api/v1/chat", json={"message": "Explain how LoRA works"})
    assert r.status_code == 200 and r.json()["provider"] == "mock"


def test_all_llms_timing_out_is_a_503_not_a_hang():
    services = make_services([hit("2106.09685v2", 0)])
    services.llm = LLMRouter([(SlowAdapter(), ProviderLimits())], first_token_timeout=0.2)
    services.graph = None
    services.__post_init__()
    with TestClient(create_app(services=services)) as client:
        client.headers.update(signup(client))
        r = client.post("/api/v1/chat", json={"message": "Explain how LoRA works"})
    assert r.status_code == 503 and "retry-after" in r.headers


def test_retriever_fake_is_still_a_retriever():
    assert isinstance(FakeRetriever(), FakeRetriever)
