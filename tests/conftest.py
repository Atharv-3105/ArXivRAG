import os

# Settings() runs at import time and requires SECRET_KEY, so env must be set before any src import.
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("BCRYPT_ROUNDS", "4")  # the minimum: hashing speed is not under test

from datetime import UTC, datetime  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.pipeline.models import PaperMetadata, SectionContent  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """App with offline fakes: mock LLM, empty retriever, in-memory history (no Qdrant/Postgres)."""
    from tests.chat_fakes import make_services

    app = create_app(services=make_services())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def make_paper():
    """Factory for PaperMetadata so tests only spell out what they care about."""

    def _make(arxiv_id: str = "2401.00001v1", **overrides) -> PaperMetadata:
        fields = {
            "arxiv_id": arxiv_id,
            "title": f"Paper {arxiv_id}",
            "authors": ["Ada Lovelace"],
            "abstract": "An abstract.",
            "published": datetime(2024, 1, 1, tzinfo=UTC),
            "updated": datetime(2024, 1, 2, tzinfo=UTC),
            "categories": ["cs.LG"],
            "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
            "primary_category": "cs.LG",
        }
        fields.update(overrides)
        return PaperMetadata(**fields)

    return _make


@pytest.fixture
def make_section():
    def _make(text: str, title: str = "Introduction", page: int = 1) -> SectionContent:
        return SectionContent(title=title, text=text, page_start=page, page_end=page)

    return _make
