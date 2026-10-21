"""Brute-force protection on sign-in and sign-up.

A gap flagged in the production review: nothing limited password guessing. Only FAILED attempts
count, per (email, IP), per IP and per email, and the check runs before the password is verified.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.limits import AuthThrottle, SlidingWindowLimiter
from src.core.config import settings
from tests.chat_fakes import make_services

PW = "correct horse"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(settings, "LOGIN_MAX_FAILURES", 4)
    monkeypatch.setattr(settings, "LOGIN_IP_MAX_FAILURES", 10)
    monkeypatch.setattr(settings, "REGISTER_MAX_PER_IP_PER_HOUR", 3)
    with TestClient(create_app(services=make_services())) as client:
        client.post("/api/v1/auth/register", json={"email": "ada@example.com", "password": PW})
        yield client


def _login(client, password, email="ada@example.com"):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def test_repeated_wrong_passwords_lock_the_account_out_for_this_address(api):
    assert [_login(api, "wrong password").status_code for _ in range(4)] == [401] * 4
    locked = _login(api, "wrong password")
    assert locked.status_code == 429
    assert locked.json()["error"]["code"] == "RATE_LIMITED"
    assert int(locked.headers["retry-after"]) > 0


def test_the_correct_password_is_refused_while_locked_out(api):
    """A guesser who happens to hit the right password mid-lockout must not get in."""
    for _ in range(4):
        _login(api, "wrong password")
    assert _login(api, PW).status_code == 429


def test_a_success_clears_the_failures(api):
    for _ in range(3):
        _login(api, "wrong password")
    assert _login(api, PW).status_code == 200
    for _ in range(3):  # three more failures are fine again: the count started over
        assert _login(api, "wrong password").status_code == 401


def test_failures_against_one_email_do_not_block_another(api):
    api.post("/api/v1/auth/register", json={"email": "bob@example.com", "password": PW})
    for _ in range(4):
        _login(api, "wrong password")
    assert _login(api, PW, email="bob@example.com").status_code == 200


def test_unknown_emails_are_throttled_too_without_revealing_that_they_do_not_exist(api):
    codes = [_login(api, "x" * 10, email="nobody@example.com").status_code for _ in range(5)]
    assert codes == [401, 401, 401, 401, 429]


def test_sign_ups_from_one_address_are_capped(api):
    # one account was already created by the fixture
    codes = [
        api.post(
            "/api/v1/auth/register", json={"email": f"u{i}@example.com", "password": PW}
        ).status_code
        for i in range(3)
    ]
    assert codes == [201, 201, 429]


# ---------------------------------------------------------------- the pieces, with a fake clock


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_peek_does_not_record_and_reset_forgets():
    clock = Clock()
    limiter = SlidingWindowLimiter(2, window=10, clock=clock)
    assert limiter.peek("k") is None
    limiter.hit("k")
    limiter.hit("k")
    assert limiter.peek("k") == pytest.approx(10)
    assert limiter.peek("k") == pytest.approx(10)  # peeking twice changed nothing
    clock.now = 10.5  # the window has passed
    assert limiter.peek("k") is None
    limiter.hit("k")
    limiter.hit("k")
    limiter.reset("k")
    assert limiter.peek("k") is None


def test_the_lockout_ends_when_the_window_does(monkeypatch):
    monkeypatch.setattr(settings, "LOGIN_MAX_FAILURES", 2)
    throttle = AuthThrottle()
    clock = Clock()
    throttle.pair = SlidingWindowLimiter(2, window=900, clock=clock)
    throttle.login_failed("a@b.c", "1.2.3.4")
    throttle.pair.hit(("a@b.c", "1.2.3.4"))
    with pytest.raises(Exception, match="Too many failed"):
        throttle.check_login("a@b.c", "1.2.3.4")
    clock.now = 901
    throttle.check_login("a@b.c", "1.2.3.4")  # no longer locked


def test_an_attacker_cannot_lock_a_victim_out_from_another_address(monkeypatch):
    monkeypatch.setattr(settings, "LOGIN_MAX_FAILURES", 3)
    throttle = AuthThrottle()
    for _ in range(3):
        throttle.login_failed("victim@example.com", "6.6.6.6")
    with pytest.raises(Exception, match="Too many failed"):
        throttle.check_login("victim@example.com", "6.6.6.6")
    throttle.check_login("victim@example.com", "10.0.0.1")  # the real owner is unaffected
