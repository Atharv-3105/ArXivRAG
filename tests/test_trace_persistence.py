"""Each turn's trace is stored with the answer, so it can be shown again after a reload."""

from fastapi.testclient import TestClient

from src.api.app import create_app
from tests.chat_fakes import hit, make_services, signup


def _client():
    services = make_services([hit("2106.09685v2", 0), hit("2305.14314v1", 0)])
    return TestClient(create_app(services=services))


def test_a_stored_answer_carries_how_it_was_made():
    with _client() as client:
        headers = signup(client)
        turn = client.post(
            "/api/v1/chat", json={"message": "Explain how LoRA works"}, headers=headers
        ).json()
        stored = client.get(
            f"/api/v1/conversations/{turn['conversation_id']}", headers=headers
        ).json()
        answer = next(m for m in stored if m["role"] == "assistant")

    trace = answer["trace"]
    assert trace["intent"]["intent"] and trace["intent"]["confidence"] is not None
    assert set(trace["timings"]) >= {"route", "retrieve", "guardrail"}
    assert all(isinstance(v, float) for v in trace["timings"].values())
    assert trace["retrieval"]["passages"] >= 1 and trace["retrieval"]["papers"] >= 1
    assert trace["guardrail"]["approved"] in (True, False)
    assert "verdict" in trace and "model" in trace


def test_the_user_message_has_no_trace():
    with _client() as client:
        headers = signup(client)
        turn = client.post("/api/v1/chat", json={"message": "hello"}, headers=headers).json()
        stored = client.get(
            f"/api/v1/conversations/{turn['conversation_id']}", headers=headers
        ).json()
    assert next(m for m in stored if m["role"] == "user").get("trace") is None


def test_the_conversation_list_keeps_titles():
    """The in-memory store (the demo, the tests) dropped titles, so History read "Untitled
    conversation" for everything."""
    with _client() as client:
        headers = signup(client)
        client.post("/api/v1/chat", json={"message": "Explain how LoRA works"}, headers=headers)
        rows = client.get("/api/v1/conversations", headers=headers).json()
    assert rows[0]["title"] == "Explain how LoRA works"
