"""run_pipeline orchestration with every collaborator faked.

What we're proving here is the *control flow contract*: counts, DLQ routing, isolation of one bad
paper from the rest, idempotency across runs, safe re-ingestion, and that errors are not swallowed.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.exceptions import FetchError, ParseError
from src.core.exceptions import IndexError as ArxivIndexError
from src.pipeline import runner
from src.pipeline.models import Chunk, Equation, ParsedDocument, SectionContent
from tests.helpers import words


class World:
    """Fakes for fetcher/extractor/chunker/stores plus a handle on what was called."""

    def __init__(self, tmp_path, papers):
        self.tmp_path = tmp_path
        self.papers = papers
        self.pg_ids: set[str] = set()  # what "Postgres" contains (stateful across runs)
        self.qdrant_ids: set[str] = set()  # what "Qdrant" contains
        self.dlq: list[tuple[str, str]] = []
        self.chunks_per_paper = 3
        self.method: dict[str, str] = {}  # arxiv_id -> "html" | "pdf" (default html)
        self.extract_errors: dict[str, Exception] = {}
        self.vector_errors: list[Exception] = []  # popped one per upsert call
        self.events: list[str] = []  # ordered log of side effects, for sequencing assertions

        self.fetcher = MagicMock()
        self.fetcher.fetch_papers = AsyncMock(side_effect=lambda **_: list(self.papers))
        self.fetcher.aclose = AsyncMock()

        self.extractor = MagicMock()
        self.extractor.extract = AsyncMock(side_effect=self._extract)

        self.chunker = MagicMock()
        self.chunker.chunk_sections = MagicMock(side_effect=self._chunk)

        self.repo = MagicMock()
        self.repo.upsert_paper_and_chunks = AsyncMock(side_effect=self._pg_upsert)
        self.repo.delete_paper = AsyncMock(side_effect=self._pg_delete)
        self.repo.log_failure_to_dlq = AsyncMock(side_effect=self._dlq)

        self.vectors = MagicMock()
        self.vectors.initialize = AsyncMock()
        self.vectors.close = AsyncMock()
        self.vectors.upsert_vectors = AsyncMock(side_effect=self._vector_upsert)
        self.vectors.delete_paper = AsyncMock(side_effect=self._vector_delete)

    # --- fake behaviours ---
    async def _extract(self, paper):
        if paper.arxiv_id in self.extract_errors:
            raise self.extract_errors[paper.arxiv_id]
        self.events.append(f"extract:{paper.arxiv_id}")
        method = self.method.get(paper.arxiv_id, "html")
        section = SectionContent(title="Introduction", text=words(300))
        return ParsedDocument(
            method=method,
            math_fidelity="latex" if method == "html" else "degraded",
            sections=[section],
            equations=[Equation(ordinal=0, latex="a=b", number="(1)")],
            stats={"words": 300},
        )

    def _chunk(self, paper_id, sections):
        return [
            Chunk(chunk_id=f"{paper_id}-{i}", paper_id=paper_id, section_title="S", content="x")
            for i in range(self.chunks_per_paper)
        ]

    async def _pg_upsert(self, session, processed):
        if processed.metadata.arxiv_id in self.pg_ids:
            return False
        self.events.append(f"pg_insert:{processed.metadata.arxiv_id}")
        self.pg_ids.add(processed.metadata.arxiv_id)
        return True

    async def _pg_delete(self, session, arxiv_id):
        self.events.append(f"pg_delete:{arxiv_id}")
        self.pg_ids.discard(arxiv_id)

    async def _vector_upsert(self, processed):
        if self.vector_errors:
            raise self.vector_errors.pop(0)
        self.events.append(f"vec_insert:{processed.metadata.arxiv_id}")
        self.qdrant_ids.add(processed.metadata.arxiv_id)

    async def _vector_delete(self, arxiv_id):
        self.events.append(f"vec_delete:{arxiv_id}")
        self.qdrant_ids.discard(arxiv_id)

    async def _dlq(self, session, arxiv_id, reason):
        self.dlq.append((arxiv_id, reason))


@pytest.fixture
def world(monkeypatch, tmp_path, make_paper):
    w = World(tmp_path, [make_paper("2401.00001v1"), make_paper("2401.00002v1")])

    @asynccontextmanager
    async def fake_session():
        yield MagicMock()

    monkeypatch.setattr(runner, "init_db_schema", AsyncMock())
    monkeypatch.setattr(runner, "dispose_engine", AsyncMock())
    monkeypatch.setattr(runner, "VectorStore", lambda *a, **k: w.vectors)
    monkeypatch.setattr(runner, "PgRepo", lambda: w.repo)
    monkeypatch.setattr(runner, "ArxivFetcher", lambda: w.fetcher)
    monkeypatch.setattr(runner, "PaperExtractor", lambda fetcher, cache: w.extractor)
    monkeypatch.setattr(runner, "TextChunker", lambda: w.chunker)
    monkeypatch.setattr(runner, "get_async_session", fake_session)
    return w


async def _run(world, **kw):
    return await runner.run_pipeline("llm", cache_dir=str(world.tmp_path / "cache"), **kw)


# ---------- happy path ----------


async def test_happy_path_counts_and_indexes_every_paper(world):
    m = await _run(world)
    assert (m.fetched, m.parsed, m.failed, m.skipped) == (2, 2, 0, 0)
    assert m.indexed == 6  # 2 papers x 3 chunks
    assert world.pg_ids == world.qdrant_ids == {"2401.00001v1", "2401.00002v1"}
    assert world.dlq == []


async def test_metrics_split_papers_by_parse_tier(world):
    world.method["2401.00002v1"] = "pdf"
    m = await _run(world)
    assert (m.html, m.pdf) == (1, 1)


async def test_equations_and_parse_provenance_reach_the_repository(world):
    await _run(world)
    processed = world.repo.upsert_paper_and_chunks.await_args_list[0].args[1]
    assert processed.parse_method == "html"
    assert processed.math_fidelity == "latex"
    assert [e.latex for e in processed.equations] == ["a=b"]
    assert processed.parse_stats == {"words": 300}


async def test_pdf_tier_papers_are_flagged_as_degraded_math(world):
    world.method["2401.00001v1"] = "pdf"
    await _run(world)
    processed = world.repo.upsert_paper_and_chunks.await_args_list[0].args[1]
    assert (processed.parse_method, processed.math_fidelity) == ("pdf", "degraded")


# ---------- failure isolation & DLQ ----------


async def test_parse_failure_goes_to_dlq_and_others_continue(world):
    world.extract_errors["2401.00001v1"] = ParseError("too short", code="PARSE_LENGTH_ERROR")
    m = await _run(world)
    assert m.failed == 1
    assert world.qdrant_ids == {"2401.00002v1"}
    assert [a for a, _ in world.dlq] == ["2401.00001v1"]
    assert "too short" in world.dlq[0][1]


async def test_download_failure_goes_to_dlq_and_others_continue(world):
    world.extract_errors["2401.00002v1"] = FetchError("404", code="FETCH_PDF_ERROR")
    m = await _run(world)
    assert m.failed == 1
    assert world.qdrant_ids == {"2401.00001v1"}
    assert [a for a, _ in world.dlq] == ["2401.00002v1"]


async def test_unexpected_exception_goes_to_dlq_marked_unknown(world):
    world.extract_errors["2401.00001v1"] = ZeroDivisionError("bug")
    m = await _run(world)
    assert m.failed == 1
    assert world.qdrant_ids == {"2401.00002v1"}
    assert world.dlq[0][1].startswith("UNKNOWN")


async def test_vector_failure_is_recorded_in_dlq(world):
    world.vector_errors = [ArxivIndexError("qdrant down", code="VECTOR_UPSERT_ERROR")]
    m = await _run(world)
    assert m.failed == 1
    assert [a for a, _ in world.dlq] == ["2401.00001v1"]


async def test_paper_with_too_few_chunks_is_skipped_and_not_stored(world):
    world.chunks_per_paper = 1
    m = await _run(world)
    assert m.skipped == 2
    assert world.pg_ids == set()
    world.repo.upsert_paper_and_chunks.assert_not_awaited()


async def test_arxiv_outage_is_not_swallowed(world):
    """If the metadata fetch itself fails, the caller (cron/Airflow) must see a failure."""
    world.fetcher.fetch_papers = AsyncMock(
        side_effect=FetchError("arXiv down", code="FETCH_API_ERROR")
    )
    with pytest.raises(FetchError):
        await _run(world)


async def test_resources_are_closed_even_when_the_run_fails(world):
    world.fetcher.fetch_papers = AsyncMock(side_effect=FetchError("down", code="FETCH_API_ERROR"))
    with pytest.raises(FetchError):
        await _run(world)
    world.fetcher.aclose.assert_awaited_once()
    world.vectors.close.assert_awaited_once()


async def test_dlq_write_failure_does_not_abort_the_run(world):
    world.extract_errors["2401.00001v1"] = ParseError("bad", code="PARSE_LENGTH_ERROR")
    world.repo.log_failure_to_dlq = AsyncMock(side_effect=RuntimeError("db down"))
    m = await _run(world)
    assert m.failed == 1
    assert world.qdrant_ids == {"2401.00002v1"}


# ---------- idempotency (plan: 'Run ingestion twice; verify zero duplicates') ----------


async def test_second_run_writes_nothing_new(world):
    await _run(world)
    vector_calls_after_first = world.vectors.upsert_vectors.await_count
    m = await _run(world)
    assert world.vectors.upsert_vectors.await_count == vector_calls_after_first
    assert m.indexed == 0
    assert len(world.pg_ids) == 2


async def test_already_stored_paper_is_counted_as_skipped(world):
    world.pg_ids.add("2401.00001v1")
    world.qdrant_ids.add("2401.00001v1")
    m = await _run(world)
    assert m.skipped == 1
    assert m.indexed == 3


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN DESIGN QUESTION: Postgres and Qdrant cannot share a transaction. If Qdrant "
        "fails after Postgres commits, the retry sees 'paper exists' and never indexes it. "
        "Decide the source of truth / reconciliation strategy, then remove this marker."
    ),
)
async def test_paper_is_indexed_on_retry_after_vector_store_failure(world):
    world.papers = world.papers[:1]
    world.vector_errors = [ArxivIndexError("qdrant down", code="VECTOR_UPSERT_ERROR")]

    await _run(world)  # Postgres commits, Qdrant fails
    await _run(world)  # operator re-runs the same ingestion

    assert world.qdrant_ids == {"2401.00001v1"}, "paper is stuck in Postgres but not searchable"


# ---------- re-ingestion (upgrade a stored paper, e.g. PDF parse -> HTML parse) ----------


async def test_reingest_replaces_stored_papers_with_the_new_parse(world):
    world.pg_ids |= {"2401.00001v1", "2401.00002v1"}
    world.qdrant_ids |= {"2401.00001v1", "2401.00002v1"}
    m = await _run(world, reingest=True)
    assert m.indexed == 6 and m.skipped == 0
    assert world.repo.delete_paper.await_count == 2
    assert world.vectors.delete_paper.await_count == 2
    assert world.pg_ids == world.qdrant_ids == {"2401.00001v1", "2401.00002v1"}


async def test_reingest_deletes_only_after_the_new_parse_succeeded(world):
    """Old data is deleted, then replaced, in that order, and only once extraction worked."""
    world.papers = world.papers[:1]
    world.pg_ids.add("2401.00001v1")
    await _run(world, reingest=True)
    order = [e for e in world.events if "2401.00001v1" in e]
    assert order == [
        "extract:2401.00001v1",
        "pg_delete:2401.00001v1",
        "vec_delete:2401.00001v1",
        "pg_insert:2401.00001v1",
        "vec_insert:2401.00001v1",
    ]


async def test_a_failed_reparse_never_destroys_the_stored_paper(world):
    world.papers = world.papers[:1]
    world.pg_ids.add("2401.00001v1")
    world.qdrant_ids.add("2401.00001v1")
    world.extract_errors["2401.00001v1"] = ParseError("bad", code="PARSE_STRUCTURE_ERROR")
    m = await _run(world, reingest=True)
    assert m.failed == 1
    world.repo.delete_paper.assert_not_awaited()
    world.vectors.delete_paper.assert_not_awaited()
    assert world.pg_ids == world.qdrant_ids == {"2401.00001v1"}  # still there


async def test_without_the_flag_nothing_is_deleted(world):
    await _run(world)
    world.repo.delete_paper.assert_not_awaited()
    world.vectors.delete_paper.assert_not_awaited()
