"""Auth, the /me surface, limits, ownership and the chat -> memory loop, through the API."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.config import settings
from tests.chat_fakes import hit, make_services, signup

PW = "correct horse"


@pytest.fixture
def services():
    return make_services([hit("2106.09685v2", 0)])


@pytest.fixture
def api(services):
    with TestClient(create_app(services=services)) as client:
        yield client


def _chat(client, headers, message="Explain how LoRA works", **extra):
    return client.post("/api/v1/chat", json={"message": message, **extra}, headers=headers)


# ---------------------------------------------------------------- auth


def test_register_then_login(api):
    r = api.post("/api/v1/auth/register", json={"email": "Ada@Example.com", "password": PW})
    assert r.status_code == 201 and r.json()["token_type"] == "bearer"
    r = api.post("/api/v1/auth/login", json={"email": "ada@example.com", "password": PW})
    assert r.status_code == 200 and r.json()["access_token"]


def test_duplicate_email_is_409(api):
    signup(api)
    r = api.post("/api/v1/auth/register", json={"email": "ADA@example.com", "password": PW})
    assert r.status_code == 409 and r.json()["error"]["code"] == "EMAIL_TAKEN"


@pytest.mark.parametrize(
    "body",
    [
        {"email": "not-an-email", "password": PW},
        {"email": "a@b.co", "password": "short"},
        {"email": "a@b.co", "password": "x" * 73},  # beyond bcrypt's 72 bytes
    ],
)
def test_bad_registration_is_400(api, body):
    assert api.post("/api/v1/auth/register", json=body).status_code == 400


def test_wrong_password_and_unknown_email_look_identical(api):
    signup(api)
    wrong = api.post(
        "/api/v1/auth/login", json={"email": "ada@example.com", "password": "nope-nope"}
    )
    unknown = api.post("/api/v1/auth/login", json={"email": "who@example.com", "password": PW})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"]
    assert wrong.headers["www-authenticate"] == "Bearer"


def test_registration_can_be_closed(api, monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_REGISTRATION", False)
    r = api.post("/api/v1/auth/register", json={"email": "a@b.co", "password": PW})
    assert r.status_code == 403


def test_admin_emails_become_admins(api, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_EMAILS", ["Boss@Example.com"])
    headers = signup(api, "boss@example.com")
    assert api.get("/api/v1/me", headers=headers).json()["is_admin"] is True


def _token(**claims):
    now = datetime.now(UTC)
    base = {"sub": "1", "iat": now, "exp": now + timedelta(hours=1), "iss": settings.APP_NAME}
    return jwt.encode({**base, **claims}, settings.SECRET_KEY, algorithm="HS256")


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({}, "AUTH_REQUIRED"),
        ({"Authorization": "Bearer garbage"}, "INVALID_TOKEN"),
        (
            {"Authorization": f"Bearer {_token(exp=datetime.now(UTC) - timedelta(seconds=5))}"},
            "TOKEN_EXPIRED",
        ),
        (
            {"Authorization": "Bearer " + jwt.encode({"sub": "1"}, "wrong-key-" * 4, "HS256")},
            "INVALID_TOKEN",
        ),
        (
            {
                "Authorization": "Bearer "
                + jwt.encode({"sub": "1", "exp": 9e9, "iat": 0}, None, "none")
            },
            "INVALID_TOKEN",
        ),  # alg=none is never accepted
    ],
)
def test_protected_routes_reject_bad_tokens(api, headers, code):
    signup(api)
    r = api.get("/api/v1/me", headers=headers)
    assert r.status_code == 401 and r.json()["error"]["code"] == code


# ---------------------------------------------------------------- profile


def test_new_user_has_default_profile(api):
    body = api.get("/api/v1/me", headers=signup(api)).json()
    assert body["profile"]["expertise_level"] == "intermediate"
    assert body["locked_fields"] == []


def test_explicit_profile_update_is_locked_and_audited(api):
    headers = signup(api)
    r = api.patch(
        "/api/v1/me/profile",
        json={"expertise_level": "beginner", "goals": ["understand RLHF"]},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["profile"]["expertise_level"] == "beginner"
    assert set(r.json()["locked_fields"]) == {"expertise_level", "goals"}
    audit = api.get("/api/v1/me/audit", headers=headers).json()
    assert {a["field"] for a in audit} >= {"expertise_level", "goals"}
    assert all(a["source"] == "explicit" for a in audit)

    r = api.patch("/api/v1/me/profile", json={"unlock": ["expertise_level"]}, headers=headers)
    assert r.json()["locked_fields"] == ["goals"]


def test_empty_or_invalid_profile_update_is_400(api):
    headers = signup(api)
    assert api.patch("/api/v1/me/profile", json={}, headers=headers).status_code == 400
    bad = api.patch("/api/v1/me/profile", json={"expertise_level": "wizard"}, headers=headers)
    assert bad.status_code == 400


def test_explicit_mastery_lock_and_unlock(api):
    headers = signup(api)
    r = api.put("/api/v1/me/mastery/lora", json={"mastery": 0.9}, headers=headers)
    assert r.status_code == 200 and r.json()["locked"] is True and r.json()["mastery"] == 0.9
    assert (
        api.put("/api/v1/me/mastery/zorblax", json={"mastery": 0.5}, headers=headers).status_code
        == 400
    )
    assert (
        api.put("/api/v1/me/mastery/lora", json={"mastery": 1.5}, headers=headers).status_code
        == 400
    )
    r = api.put("/api/v1/me/mastery/lora", json={"mastery": None}, headers=headers)
    assert r.json()["locked"] is False


# ---------------------------------------------------------------- chat -> memory (async)


def test_chat_turn_updates_mastery_asynchronously(api, services):
    headers = signup(api)
    r = _chat(api, headers)
    assert r.status_code == 200 and r.json()["concepts"] == ["lora"]
    api.portal.call(services.memory.drain)  # wait for the background worker
    [row] = api.get("/api/v1/me/mastery", headers=headers).json()
    assert row["concept"] == "lora" and row["evidence"] == 1
    assert row["mastery"] < 0.45  # "explain X" is weak evidence of a gap
    audit = api.get("/api/v1/me/audit", headers=headers).json()
    assert audit[0]["field"] == "mastery:lora" and audit[0]["source"] == "inferred"


def test_locked_concept_is_not_changed_by_chat(api, services):
    headers = signup(api)
    api.put("/api/v1/me/mastery/lora", json={"mastery": 0.95}, headers=headers)
    _chat(api, headers)
    api.portal.call(services.memory.drain)
    [row] = api.get("/api/v1/me/mastery", headers=headers).json()
    assert row["mastery"] == 0.95


def test_feedback_is_accepted_async_and_moves_mastery(api, services):
    headers = signup(api)
    r = api.post(
        "/api/v1/feedback",
        json={"signal": "confused", "concepts": ["lora", "not_a_concept"]},
        headers=headers,
    )
    assert r.status_code == 202 and r.json() == {"queued": True, "concepts": ["lora"]}
    api.portal.call(services.memory.drain)
    [row] = api.get("/api/v1/me/mastery", headers=headers).json()
    assert row["mastery"] < 0.45


def test_due_reviews(api, services):
    headers = signup(api)
    _chat(api, headers)
    api.portal.call(services.memory.drain)
    assert api.get("/api/v1/me/reviews", headers=headers).json() == []  # due tomorrow
    state = services.profiles.mastery[1]["lora"]
    state.next_review_at = datetime.now(UTC) - timedelta(minutes=1)
    [due] = api.get("/api/v1/me/reviews", headers=headers).json()
    assert due["concept"] == "lora"


def test_profile_reaches_the_prompt(api, services):
    headers = signup(api)
    api.patch("/api/v1/me/profile", json={"expertise_level": "beginner"}, headers=headers)
    api.put("/api/v1/me/mastery/lora", json={"mastery": 0.1}, headers=headers)
    _chat(api, headers)
    system = services.llm.slots[0].adapter.requests[-1].messages[0].content
    assert "The reader is a beginner" in system
    assert "Still shaky on: lora" in system


# ---------------------------------------------------------------- limits


def test_rate_limit_returns_429_with_retry_after(monkeypatch):
    services = make_services([hit("p", 0)])
    services.limiter.limit = 2
    with TestClient(create_app(services=services)) as client:
        headers = signup(client)
        codes = [_chat(client, headers).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = _chat(client, headers)
        assert int(r.headers["retry-after"]) >= 1 and r.json()["error"]["code"] == "RATE_LIMITED"


def test_daily_quota_survives_in_the_message_store(monkeypatch):
    monkeypatch.setattr(settings, "FREE_DAILY_MESSAGES", 2)
    with TestClient(create_app(services=make_services([hit("p", 0)]))) as client:
        headers = signup(client)
        _chat(client, headers)
        _chat(client, headers)
        r = _chat(client, headers)
        assert r.status_code == 429 and r.json()["error"]["code"] == "DAILY_QUOTA_EXCEEDED"


def test_admins_are_exempt_from_limits(monkeypatch):
    monkeypatch.setattr(settings, "FREE_DAILY_MESSAGES", 1)
    monkeypatch.setattr(settings, "ADMIN_EMAILS", ["root@example.com"])
    with TestClient(create_app(services=make_services([hit("p", 0)]))) as client:
        headers = signup(client, "root@example.com")
        assert [_chat(client, headers).status_code for _ in range(3)] == [200, 200, 200]


# ---------------------------------------------------------------- ownership & GDPR


def test_another_users_conversation_is_invisible(api):
    alice, bob = signup(api, "alice@example.com"), signup(api, "bob@example.com")
    cid = _chat(api, alice).json()["conversation_id"]
    assert _chat(api, bob, conversation_id=cid).status_code == 404
    assert api.get(f"/api/v1/conversations/{cid}", headers=bob).status_code == 404
    assert api.get(f"/api/v1/conversations/{cid}", headers=alice).status_code == 200
    assert [c["id"] for c in api.get("/api/v1/conversations", headers=alice).json()] == [cid]
    assert api.get("/api/v1/conversations", headers=bob).json() == []


def test_export_contains_everything(api, services):
    headers = signup(api)
    _chat(api, headers)
    api.portal.call(services.memory.drain)
    data = api.get("/api/v1/me/export", headers=headers).json()
    assert data["account"]["email"] == "ada@example.com"
    assert data["mastery"][0]["concept"] == "lora"
    assert data["events"][0]["kind"] == "chat_turn"
    assert data["audit"] and len(data["conversations"][0]["messages"]) == 2


def test_account_deletion_requires_the_password_and_erases_everything(api, services):
    headers = signup(api)
    _chat(api, headers)
    api.portal.call(services.memory.drain)
    wrong = api.request("DELETE", "/api/v1/me", json={"password": "wrong-pass"}, headers=headers)
    assert wrong.status_code == 401
    ok = api.request("DELETE", "/api/v1/me", json={"password": PW}, headers=headers)
    assert ok.status_code == 204
    assert api.get("/api/v1/me", headers=headers).status_code == 401  # token dies with the account
    assert services.conversations.owners == {} and services.profiles.mastery == {}


# ---------------------------------------------------------------- profile text is bounded


def test_profile_goals_and_interests_are_bounded(api):
    """A 1 MB goal was accepted, and these strings are injected into every tutor prompt."""
    headers = signup(api)
    for body in (
        {"goals": ["x" * 201]},
        {"interests": ["y" * 61]},
        {"goals": [""]},
        {"goals": ["   "]},
    ):
        r = api.patch("/api/v1/me/profile", json=body, headers=headers)
        assert r.status_code == 400, body
    ok = api.patch(
        "/api/v1/me/profile",
        json={"goals": ["  understand RLHF  "], "interests": ["alignment"]},
        headers=headers,
    )
    assert ok.status_code == 200 and ok.json()["profile"]["goals"] == ["understand RLHF"]


def test_feedback_concepts_are_bounded(api):
    headers = signup(api)
    r = api.post(
        "/api/v1/feedback", json={"signal": "helpful", "concepts": ["q" * 61]}, headers=headers
    )
    assert r.status_code == 400
