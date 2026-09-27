"""Owner admin account seeded from the environment at startup."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api.app import create_app
from src.auth.security import hash_password
from src.auth.seed import ensure_admin
from src.auth.users import InMemoryUserStore
from src.core.config import settings
from tests.chat_fakes import make_services

EMAIL = "Owner@Example.com"
PASSWORD = "a-long-test-password-not-for-production"


def _configure(monkeypatch, email=EMAIL, password=PASSWORD):
    monkeypatch.setattr(settings, "SEED_ADMIN_EMAIL", email)
    monkeypatch.setattr(settings, "SEED_ADMIN_PASSWORD", SecretStr(password) if password else None)


async def test_creates_the_admin_once(monkeypatch):
    _configure(monkeypatch)
    users = InMemoryUserStore()
    assert await ensure_admin(users) == "created"
    assert await ensure_admin(users) == "exists"  # idempotent across restarts
    user = await users.by_email("owner@example.com")
    assert user.is_admin and len(users.users) == 1


async def test_promotes_an_existing_account_without_touching_its_password(monkeypatch):
    _configure(monkeypatch)
    users = InMemoryUserStore()
    original = hash_password("the-users-own-password")
    await users.create("owner@example.com", original, None, False)
    assert await ensure_admin(users) == "promoted"
    user = await users.by_email("owner@example.com")
    assert user.is_admin and user.password_hash == original


async def test_skipped_without_email_or_valid_password(monkeypatch):
    users = InMemoryUserStore()
    _configure(monkeypatch, email=None)
    assert await ensure_admin(users) == "skipped"
    _configure(monkeypatch, password="short")
    assert await ensure_admin(users) == "skipped"
    _configure(monkeypatch, password=None)
    assert await ensure_admin(users) == "skipped"
    assert users.users == {}


async def test_seeded_admin_can_sign_in_and_is_exempt_from_limits(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(settings, "FREE_DAILY_MESSAGES", 1)
    services = make_services()
    await ensure_admin(services.users)
    with TestClient(create_app(services=services)) as client:
        r = client.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
        assert r.status_code == 200
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        assert client.get("/api/v1/me", headers=headers).json()["is_admin"] is True
        codes = [
            client.post("/api/v1/chat", json={"message": "hello"}, headers=headers).status_code
            for _ in range(3)
        ]
        assert codes == [200, 200, 200]  # daily quota of 1 does not apply to admins
