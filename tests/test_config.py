"""Phase 0 acceptance: 'config validation fails on missing env vars'."""

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def test_missing_secret_key_fails_at_startup(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)
    assert "SECRET_KEY" in str(exc.value)


def test_defaults_are_valid():
    s = Settings(_env_file=None, SECRET_KEY="x")
    assert s.PORT == 8000
    assert isinstance(s.PORT, int)
    assert s.LOG_LEVEL == "INFO"


def test_invalid_log_level_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, SECRET_KEY="x", LOG_LEVEL="loud")


def test_log_level_is_normalised_to_upper_case():
    assert Settings(_env_file=None, SECRET_KEY="x", LOG_LEVEL="debug").LOG_LEVEL == "DEBUG"


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("PORT", "9001")
    assert Settings(_env_file=None).PORT == 9001


def test_plain_postgres_url_is_upgraded_to_the_async_driver():
    s = Settings(_env_file=None, SECRET_KEY="x", DATABASE_URL="postgresql://u:p@localhost:5433/db")
    assert s.DATABASE_URL == "postgresql+asyncpg://u:p@localhost:5433/db"


def test_async_postgres_url_is_left_alone():
    url = "postgresql+asyncpg://u:p@localhost:5433/db"
    assert Settings(_env_file=None, SECRET_KEY="x", DATABASE_URL=url).DATABASE_URL == url


def test_default_arxiv_query_uses_field_prefixes_not_bare_words():
    q = Settings(_env_file=None, SECRET_KEY="x").ARXIV_QUERY
    assert "cat:" in q and 'abs:"' in q
