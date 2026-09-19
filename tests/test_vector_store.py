"""VectorStore with a mocked Qdrant client and a fake embedder (no model download, no server)."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.core.exceptions import IndexError as ArxivIndexError
from src.pipeline.chunker import TextChunker
from src.pipeline.models import ProcessedPaper, SectionContent
from src.storage import vector_db
from src.storage.embedder import EmbeddingService
from tests.helpers import words

DIM = 384


def _is_valid_qdrant_id(value) -> bool:
    """Qdrant point ids must be an unsigned integer or a UUID. Arbitrary strings are rejected."""
    if isinstance(value, int):
        return value >= 0
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


@pytest.fixture
def store(monkeypatch):
    client = MagicMock()
    client.upsert = AsyncMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    monkeypatch.setattr(vector_db, "AsyncQdrantClient", lambda **_: client)

    embedder = MagicMock()
    embedder.encode.side_effect = lambda texts: np.ones((len(texts), DIM), dtype=np.float32)
    monkeypatch.setattr(vector_db, "EmbeddingService", lambda: embedder)

    return vector_db.VectorStore(), client, embedder


@pytest.fixture
def processed(make_paper):
    """Chunks come from the real chunker, so the chunker->vector-store id contract is exercised.

    Whether ids become Qdrant-valid in the chunker or the store is a design choice; either passes.
    """
    paper = make_paper("2401.00001v1")
    section = SectionContent(title="Introduction", text=words(300), page_start=1, page_end=1)
    chunks = TextChunker().chunk_sections(paper.arxiv_id, [section])
    assert len(chunks) >= 2, "fixture needs several chunks"
    return ProcessedPaper(metadata=paper, sections=[section], chunks=chunks)


# ---------- collection lifecycle ----------


async def test_initialize_creates_collection_when_missing(store):
    vs, client, _ = store
    client.collection_exists.return_value = False
    await vs.initialize()
    client.create_collection.assert_awaited_once()
    cfg = client.create_collection.await_args.kwargs["vectors_config"]
    assert cfg.size == DIM


async def test_initialize_is_idempotent_when_collection_exists(store):
    vs, client, _ = store
    await vs.initialize()
    client.create_collection.assert_not_awaited()


# ---------- upsert ----------


async def test_upsert_sends_one_point_per_chunk_with_provenance(store, processed):
    vs, client, _ = store
    await vs.upsert_vectors(processed)
    points = client.upsert.await_args.kwargs["points"]
    n = len(processed.chunks)
    assert len(points.ids) == len(points.vectors) == len(points.payloads) == n
    assert points.payloads[0]["paper_id"] == "2401.00001v1"
    assert points.payloads[0]["section"] == "Introduction"
    assert points.payloads[0]["content"] == processed.chunks[0].content


async def test_point_ids_are_valid_for_qdrant(store, processed):
    """Qdrant rejects arbitrary string ids like '2401.00001v1_Introduction_0' at runtime."""
    vs, client, _ = store
    await vs.upsert_vectors(processed)
    ids = client.upsert.await_args.kwargs["points"].ids
    assert all(_is_valid_qdrant_id(i) for i in ids), ids


async def test_point_ids_are_stable_across_reingestion(store, processed):
    """Same chunk -> same point id, so a re-upsert overwrites instead of duplicating."""
    vs, client, _ = store
    await vs.upsert_vectors(processed)
    await vs.upsert_vectors(processed)
    first, second = (c.kwargs["points"].ids for c in client.upsert.await_args_list)
    assert first == second


async def test_upsert_waits_for_durability(store, processed):
    vs, client, _ = store
    await vs.upsert_vectors(processed)
    assert client.upsert.await_args.kwargs["wait"] is True


async def test_empty_paper_does_not_call_qdrant(store, make_paper):
    vs, client, embedder = store
    await vs.upsert_vectors(ProcessedPaper(metadata=make_paper(), sections=[], chunks=[]))
    client.upsert.assert_not_awaited()
    embedder.encode.assert_not_called()


async def test_qdrant_failure_is_wrapped_as_index_error(store, processed):
    vs, client, _ = store
    client.upsert.side_effect = RuntimeError("qdrant down")
    with pytest.raises(ArxivIndexError) as exc:
        await vs.upsert_vectors(processed)
    assert exc.value.code == "VECTOR_UPSERT_ERROR"


async def test_embedding_count_mismatch_is_rejected(store, processed):
    vs, client, embedder = store
    embedder.encode.side_effect = lambda texts: np.ones((1, DIM))
    with pytest.raises(ArxivIndexError) as exc:
        await vs.upsert_vectors(processed)
    assert exc.value.code == "VECTOR_DIM_ERROR"
    client.upsert.assert_not_awaited()


# ---------- embedder ----------


def test_embedder_model_is_loaded_lazily():
    assert EmbeddingService()._model is None


def test_embedder_returns_empty_array_for_no_input_without_loading_model():
    svc = EmbeddingService()
    assert svc.encode([]).size == 0
    assert svc._model is None
