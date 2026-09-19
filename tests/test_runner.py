"""run_pipeline orchestration with every collaborator faked.

What we're proving here is the *control flow contract*: counts, DLQ routing, isolation of one bad
paper from the rest, idempotency across runs, and that errors are not swallowed.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.exceptions import FetchError, ParseError
from src.core.exceptions import IndexError as ArxivIndexError
from src.pipeline import runner
from src.pipeline.models import Chunk, SectionContent
from src.pipeline.parser import ParseMetrics
from tests.helpers import words


class World:
    """Fakes for fetcher/parser/chunker/stores plus a handle on what was called."""

    def __init__(self, tmp_path, papers):
        self.tmp_path = tmp_path
        self.papers = papers
        self.pg_ids: set[str] = set()  # what "Postgres" contains (stateful across runs)
        self.qdrant_ids: set[str] = set()  # what "Qdrant" contains
        self.dlq: list[tuple[str, str]] = []
        self.chunks_per_paper = 3
        self.parse_errors: dict[str, Exception] = {}
        self.download_errors: dict[str, Exception] = {}
        self.vector_errors: list[Exception] = []  # popped one per upsert call

        self.fetcher = MagicMock()
        self.fetcher.fetch_papers = AsyncMock(side_effect=lambda **_: list(self.papers))
        self.fetcher.download_pdf = AsyncMock(side_effect=self._download)
        self.fetcher.aclose = AsyncMock()

        self.parser = MagicMock()
        self.parser.parse = MagicMock(side_effect=self._parse)

        self.chunker = MagicMock()
        self.chunker.chunk_sections = MagicMock(side_effect=self._chunk)

        self.repo = MagicMock()
        self.repo.upsert_paper_and_chunks = AsyncMock(side_effect=self._pg_upsert)
        self.repo.log_failure_to_dlq = AsyncMock(side_effect=self._dlq)

        self.vectors = MagicMock()
        self.vectors.initialize = AsyncMock()
        self.vectors.close = AsyncMock()
        self.vectors.upsert_vectors = AsyncMock(side_effect=self._vector_upsert)

    # --- fake behaviours ---
    def _download(self, paper, path):
        if paper.arxiv_id in self.download_errors:
            raise self.download_errors[paper.arxiv_id]
        with open(path, "wb") as f:
            f.write(b"%PDF fake")
        return True

    def _parse(self, path):
        arxiv_id = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].removesuffix(".pdf")
        if arxiv_id in self.parse_errors:
            raise self.parse_errors[arxiv_id]
        section = SectionContent(title="Introduction", text=words(300), page_start=1, page_end=1)
        return [section], ParseMetrics()

    def _chunk(self, paper_id, sections):
        return [
            Chunk(chunk_id=f"{paper_id}-{i}", paper_id=paper_id, section_title="S", content="x")
            for i in range(self.chunks_per_paper)
        ]

    async def _pg_upsert(self, session, processed):
        if processed.metadata.arxiv_id in self.pg_ids:
            return False
        self.pg_ids.add(processed.metadata.arxiv_id)
        return True

    async def _vector_upsert(self, processed):
        if self.vector_errors:
            raise self.vector_errors.pop(0)
        self.qdrant_ids.add(processed.metadata.arxiv_id)

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
    monkeypatch.setattr(runner, "PDFParser", lambda: w.parser)
    monkeypatch.setattr(runner, "TextChunker", lambda: w.chunker)
    monkeypatch.setattr(runner, "get_async_session", fake_session)
    return w


async def _run(world, **kw):
    return await runner.run_pipeline("llm", output_dir=str(world.tmp_path / "pdfs"), **kw)


# ---------- happy path ----------


async def test_happy_path_counts_and_indexes_every_paper(world):
    m = await _run(world)
    assert (m.fetched, m.parsed, m.failed, m.skipped) == (2, 2, 0, 0)
    assert m.indexed == 6  # 2 papers x 3 chunks
    assert world.pg_ids == world.qdrant_ids == {"2401.00001v1", "2401.00002v1"}
    assert world.dlq == []


async def test_existing_pdf_is_not_downloaded_again(world, tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "2401.00001v1.pdf").write_bytes(b"%PDF cached")
    await _run(world)
    downloaded = [c.args[0].arxiv_id for c in world.fetcher.download_pdf.await_args_list]
    assert downloaded == ["2401.00002v1"]


# ---------- failure isolation & DLQ ----------


async def test_parse_failure_goes_to_dlq_and_others_continue(world):
    world.parse_errors["2401.00001v1"] = ParseError("too short", code="PARSE_LENGTH_ERROR")
    m = await _run(world)
    assert m.failed == 1
    assert world.qdrant_ids == {"2401.00002v1"}
    assert [a for a, _ in world.dlq] == ["2401.00001v1"]
    assert "too short" in world.dlq[0][1]


async def test_download_failure_goes_to_dlq_and_others_continue(world):
    world.download_errors["2401.00002v1"] = FetchError("404", code="FETCH_PDF_ERROR")
    m = await _run(world)
    assert m.failed == 1
    assert world.qdrant_ids == {"2401.00001v1"}
    assert [a for a, _ in world.dlq] == ["2401.00002v1"]


async def test_unexpected_exception_goes_to_dlq_marked_unknown(world):
    world.parse_errors["2401.00001v1"] = ZeroDivisionError("bug")
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


async def test_old_style_arxiv_id_is_safe_as_a_filename(world, make_paper):
    world.papers = [make_paper("hep-th/9901001v1")]
    await _run(world)
    saved = world.fetcher.download_pdf.await_args.args[1]
    assert saved.endswith("hep-th_9901001v1.pdf")


async def test_dlq_write_failure_does_not_abort_the_run(world):
    world.parse_errors["2401.00001v1"] = ParseError("bad", code="PARSE_LENGTH_ERROR")
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
