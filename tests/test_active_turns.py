"""One reply at a time per conversation.

Measured on the real stack: one tutor answer sent twice at once was graded `incorrect` for one
request and `partial` for the other, counted twice toward mastery, and only one state write
survived.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.limits import ActiveTurns
from tests.chat_fakes import hit, make_services, signup


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_a_conversation_can_be_claimed_once_until_released():
    turns = ActiveTurns()
    assert turns.claim("c1") is True
    assert turns.claim("c1") is False
    assert turns.claim("c2") is True  # other conversations are independent
    turns.release("c1")
    assert turns.claim("c1") is True
    turns.release("never-claimed")  # releasing twice or blindly is harmless


def test_a_claim_expires_so_a_vanished_client_cannot_lock_a_conversation():
    clock = Clock()
    turns = ActiveTurns(ttl=60, clock=clock)
    assert turns.claim("c1")
    clock.now = 59
    assert turns.claim("c1") is False
    clock.now = 61
    assert turns.claim("c1") is True


@pytest.fixture
def api():
    services = make_services([hit("2106.09685v2", 0)])
    with TestClient(create_app(services=services)) as client:
        client.services = services
        yield client


def _first_turn(api, headers):
    return api.post(
        "/api/v1/chat", json={"message": "Explain how LoRA works"}, headers=headers
    ).json()


def test_a_second_simultaneous_turn_is_refused_with_a_clear_error(api):
    headers = signup(api)
    cid = _first_turn(api, headers)["conversation_id"]
    api.services.active_turns.claim(cid)  # a reply is being written

    plain = api.post(
        "/api/v1/chat", json={"message": "hello", "conversation_id": cid}, headers=headers
    )
    stream = api.post(
        "/api/v1/chat/stream", json={"message": "hello", "conversation_id": cid}, headers=headers
    )
    for r in (plain, stream):
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "TURN_IN_PROGRESS"


def test_finished_turns_release_the_conversation(api):
    headers = signup(api)
    cid = _first_turn(api, headers)["conversation_id"]
    body = {"message": "and what about QLoRA?", "conversation_id": cid}
    assert api.post("/api/v1/chat", json=body, headers=headers).status_code == 200
    with api.stream("POST", "/api/v1/chat/stream", json=body, headers=headers) as r:
        assert r.status_code == 200
        r.read()
    assert api.post("/api/v1/chat", json=body, headers=headers).status_code == 200
    assert api.services.active_turns.claim(cid) is True  # nothing left claimed


def test_a_failed_turn_releases_the_conversation(api, monkeypatch):
    headers = signup(api)
    cid = _first_turn(api, headers)["conversation_id"]

    async def boom(_input):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(api.services.graph, "ainvoke", boom)
    with pytest.raises(RuntimeError):
        api.post("/api/v1/chat", json={"message": "x", "conversation_id": cid}, headers=headers)
    assert api.services.active_turns.claim(cid) is True


def test_another_users_turn_does_not_collide_with_mine(api):
    mine = signup(api, "me@example.com")
    theirs = signup(api, "them@example.com")
    a = _first_turn(api, mine)["conversation_id"]
    b = _first_turn(api, theirs)["conversation_id"]
    api.services.active_turns.claim(a)
    ok = api.post("/api/v1/chat", json={"message": "hi", "conversation_id": b}, headers=theirs)
    assert ok.status_code == 200
