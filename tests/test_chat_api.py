import json
import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from tests.chat_fakes import RETRIEVAL_DOWN, DeadAdapter, hit, make_services, signup


@pytest.fixture
def services():
    return make_services([hit("2106.09685v2", 0), hit("2305.14314v1", 1)])


@pytest.fixture
def api(services):
    with TestClient(create_app(services=services)) as client:
        client.headers.update(signup(client))
        yield client


def _authed(services):
    """A signed-in client for a one-off app (use as a context manager)."""

    class _Ctx:
        def __enter__(self):
            self.cm = TestClient(create_app(services=services))
            client = self.cm.__enter__()
            client.headers.update(signup(client))
            return client

        def __exit__(self, *exc):
            return self.cm.__exit__(*exc)

    return _Ctx()


def _events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_chat_returns_a_grounded_answer_with_citations(api):
    r = api.post("/api/v1/chat", json={"message": "Explain how LoRA works"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "learn"
    assert body["citations"][0]["paper_id"] == "2106.09685v2"
    assert body["citations"][0]["url"].startswith("https://arxiv.org/html/")
    assert body["trace_id"] == r.headers["x-trace-id"]
    uuid.UUID(body["conversation_id"])


def test_conversation_history_is_kept_between_turns(api, services):
    first = api.post("/api/v1/chat", json={"message": "Explain QLoRA"}).json()
    cid = first["conversation_id"]
    api.post("/api/v1/chat", json={"message": "and its limits?", "conversation_id": cid})
    stored = services.conversations.data[cid]  # user-scoped store
    assert [m["role"] for m in stored] == ["user", "assistant", "user", "assistant"]
    assert stored[1]["citations"] and stored[1]["intent"] == "learn"
    last_prompt = services.llm.slots[0].adapter.requests[-1].messages
    assert [m.role for m in last_prompt] == ["system", "user", "assistant", "user"]


def test_stream_protocol(api):
    r = api.post("/api/v1/chat/stream", json={"message": "Explain how LoRA works"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    names = [name for name, _ in events]
    assert names[:4] == ["meta", "intent", "sources", "provider"]
    assert names[-1] == "done" and "token" in names
    streamed = "".join(data["text"] for name, data in events if name == "token")
    done = events[-1][1]
    assert done["answer"] == streamed
    assert done["citations"] and set(done["timings"]) == {"classify", "retrieve", "generate"}


def test_stream_reports_llm_outage_as_an_error_event():
    services = make_services([hit("p", 0)], adapter=DeadAdapter())
    with _authed(services) as client:
        events = _events(client.post("/api/v1/chat/stream", json={"message": "Explain LoRA"}))
    name, data = events[-1]
    assert name == "error" and data["code"] == "LLM_UNAVAILABLE"
    assert data["retry_after"] >= 1
    assert services.conversations.data == {}  # a failed turn is not saved


def test_chat_maps_llm_outage_to_503_with_retry_after():
    services = make_services([hit("p", 0)], adapter=DeadAdapter())
    with _authed(services) as client:
        r = client.post("/api/v1/chat", json={"message": "Explain LoRA"})
    assert r.status_code == 503
    assert int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["code"] == "LLM_UNAVAILABLE"
    assert r.json()["error"]["trace_id"] == r.headers["x-trace-id"]


def test_retrieval_outage_degrades_instead_of_failing():
    with _authed(make_services(retrieval_error=RETRIEVAL_DOWN)) as client:
        body = client.post("/api/v1/chat", json={"message": "Explain LoRA"}).json()
    assert body["degraded"] == "retrieval_unavailable" and body["provider"] == "none"


@pytest.mark.parametrize(
    "payload",
    [{}, {"message": ""}, {"message": "x" * 5000}, {"message": "hi", "conversation_id": "nope"}],
)
def test_invalid_requests_are_400_with_details(api, payload):
    r = api.post("/api/v1/chat", json=payload)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "VALIDATION_ERROR" and r.json()["error"]["details"]


def test_history_outage_does_not_block_the_answer(services):
    async def broken(*args, **kwargs):
        raise ConnectionError("postgres down")

    services.conversations.history = broken
    services.conversations.append = broken
    services.conversations.count_user_messages_since = broken  # quota check fails open
    with _authed(services) as client:
        r = client.post("/api/v1/chat", json={"message": "Explain LoRA"})
    assert r.status_code == 200 and r.json()["answer"]
