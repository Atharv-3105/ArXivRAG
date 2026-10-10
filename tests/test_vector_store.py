"""VectorStore with a mocked Qdrant client and a fake embedder (no model download, no server)."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.core.exceptions import IndexError as ArxivIndexError
from src.core.exceptions import RetrievalError
from src.pipeline.chunker import TextChunker
from src.pipeline.models import ProcessedPaper, SectionContent
from src.storage import vector_db
from src.storage.embedder import EmbeddingService, SparseVector
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


def _collection_info(dense_size: int | None = DIM, sparse: bool = True):
    """Shape of qdrant's get_collection() response, only the fields VectorStore reads."""
    from qdrant_client.http import models

    params = MagicMock()
    params.vectors = (
        {"dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE)}
        if dense_size
        else models.VectorParams(size=DIM, distance=models.Distance.COSINE)  # old unnamed layout
    )
    params.sparse_vectors = {"bm25": models.SparseVectorParams()} if sparse else None
    info = MagicMock()
    info.config.params = params
    return info


@pytest.fixture
def store(monkeypatch):
    client = MagicMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.get_collection = AsyncMock(return_value=_collection_info())
    client.query_points = AsyncMock(return_value=MagicMock(points=[]))
    monkeypatch.setattr(vector_db, "AsyncQdrantClient", lambda **_: client)

    embedder = MagicMock()
    embedder.encode.side_effect = lambda texts: np.ones((len(texts), DIM), dtype=np.float32)
    embedder.encode_sparse.side_effect = lambda texts: (
        [SparseVector([1, 7], [0.5, 1.0])] * len(texts)
    )
    embedder.encode_query.return_value = np.ones(DIM, dtype=np.float32)
    embedder.encode_sparse_query.return_value = SparseVector([7], [1.0])
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
    kwargs = client.create_collection.await_args.kwargs
    assert kwargs["vectors_config"]["dense"].size == DIM
    assert kwargs["sparse_vectors_config"]["bm25"].modifier.value == "idf"  # IDF applied by Qdrant
    indexed = {c.kwargs["field_name"] for c in client.create_payload_index.await_args_list}
    assert {"paper_id", "metadata.math_fidelity"} <= indexed


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
    assert len(points.ids) == len(points.payloads) == n
    assert len(points.vectors["dense"]) == len(points.vectors["bm25"]) == n
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


# ---------- new in the HTML/math pipeline


async def test_chunks_are_embedded_together_with_their_section_path(store, processed):
    vs, _, embedder = store
    await vs.upsert_vectors(processed)
    texts = embedder.encode.call_args.args[0]
    assert texts == [c.embedding_text for c in processed.chunks]
    assert all(t.startswith("Introduction\n") for t in texts)  # context first, then the chunk text


async def test_payload_records_how_the_paper_was_parsed(store, processed):
    vs, client, _ = store
    processed.parse_method = "html"
    processed.math_fidelity = "latex"
    await vs.upsert_vectors(processed)
    meta = client.upsert.await_args.kwargs["points"].payloads[0]["metadata"]
    assert (meta["parse_method"], meta["math_fidelity"]) == ("html", "latex")
    assert meta["section_path"] == "Introduction"  # chunk metadata is preserved alongside


async def test_delete_paper_filters_on_paper_id(store):
    vs, client, _ = store
    await vs.delete_paper("2401.00001v1")
    selector = client.delete.await_args.kwargs["points_selector"]
    condition = selector.filter.must[0]
    assert (condition.key, condition.match.value) == ("paper_id", "2401.00001v1")


async def test_delete_failure_is_wrapped_as_index_error(store):
    vs, client, _ = store
    client.delete.side_effect = RuntimeError("qdrant down")
    with pytest.raises(ArxivIndexError) as exc:
        await vs.delete_paper("2401.00001v1")
    assert exc.value.code == "VECTOR_DELETE_ERROR"


# ---------- hybrid layout, schema guard, search


@pytest.mark.parametrize(
    "info",
    [
        _collection_info(dense_size=None),  # old dense-only, unnamed vector (v1)
        _collection_info(sparse=False),  # no BM25
        _collection_info(dense_size=768),  # different embedding model
    ],
)
async def test_existing_collection_with_another_layout_is_refused(store, info):
    vs, client, _ = store
    client.get_collection.return_value = info
    with pytest.raises(ArxivIndexError) as exc:
        await vs.initialize()
    assert exc.value.code == "VECTOR_SCHEMA_MISMATCH"
    assert "reindex --all" in exc.value.message  # the error says how to fix it


async def test_payload_carries_title_so_results_need_no_database_read(store, processed):
    vs, client, _ = store
    await vs.upsert_vectors(processed)
    payload = client.upsert.await_args.kwargs["points"].payloads[0]
    assert payload["title"] == processed.metadata.title
    assert payload["published"] == "2024-01-01"


async def test_search_fuses_dense_and_bm25_with_rrf(store):
    from qdrant_client.http import models

    vs, client, _ = store
    point = MagicMock(id="abc", score=0.5)
    point.payload = {
        "paper_id": "2401.00001v1",
        "title": "T",
        "section": "Intro",
        "content": "text",
        "metadata": {"math_fidelity": "latex"},
    }
    client.query_points.return_value = MagicMock(points=[point])

    [hit] = await vs.search("what is LoRA", top_k=4)

    kwargs = client.query_points.await_args.kwargs
    assert kwargs["query"].fusion == models.Fusion.RRF
    assert [p.using for p in kwargs["prefetch"]] == ["dense", "bm25"]
    assert kwargs["limit"] == 4
    assert (hit.paper_id, hit.title, hit.section, hit.score) == ("2401.00001v1", "T", "Intro", 0.5)


async def test_search_can_be_restricted_to_papers(store):
    vs, client, _ = store
    await vs.search("q", paper_ids=["2401.00001v1"])
    for prefetch in client.query_points.await_args.kwargs["prefetch"]:
        assert prefetch.filter.must[0].match.any == ["2401.00001v1"]


async def test_search_failure_is_a_retrieval_error(store):
    vs, client, _ = store
    client.query_points.side_effect = RuntimeError("timeout")
    with pytest.raises(RetrievalError) as exc:
        await vs.search("q")
    assert exc.value.code == "VECTOR_SEARCH_ERROR"


def _embedding_service_with_fake_model():
    svc = EmbeddingService()
    fake = MagicMock()
    fake.embed.side_effect = lambda texts, **_: [np.ones(DIM)] * len(texts)
    svc._model = fake
    return svc, fake


def test_query_embedding_sends_the_configured_instruction_prefix(monkeypatch):
    monkeypatch.setattr(vector_db.settings, "EMBEDDING_QUERY_PREFIX", "Represent this sentence: ")
    svc, fake = _embedding_service_with_fake_model()
    vec = svc.encode_query("attention")
    assert fake.embed.call_args.args[0][0] == "Represent this sentence: attention"
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6  # normalised


def test_minilm_default_sends_the_query_unchanged():
    """MiniLM is symmetric: an instruction prefix would only add noise to the query."""
    svc, fake = _embedding_service_with_fake_model()
    svc.encode_query("attention")
    assert fake.embed.call_args.args[0][0] == "attention"


async def test_hybrid_search_degrades_to_bm25_when_the_dense_model_fails(store):
    vs, client, embedder = store
    embedder.encode_query.side_effect = RuntimeError("model download blocked")
    await vs.search("LoRA rank")
    kwargs = client.query_points.await_args.kwargs
    assert kwargs["using"] == vector_db.SPARSE  # one lexical branch, no dense, no fusion
    assert "prefetch" not in kwargs


async def test_dense_only_mode_does_not_hide_a_model_failure(store):
    vs, _, embedder = store
    embedder.encode_query.side_effect = RuntimeError("model download blocked")
    with pytest.raises(RetrievalError) as exc:
        await vs.search("q", mode="dense")
    assert exc.value.code == "EMBEDDING_UNAVAILABLE"


def test_sparse_encoder_returns_empty_list_without_loading_model():
    svc = EmbeddingService()
    assert svc.encode_sparse([]) == []
    assert svc._sparse is None
