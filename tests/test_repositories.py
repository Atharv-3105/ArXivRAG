"""PgRepo with a mocked AsyncSession. Statements are compiled against the PostgreSQL dialect so we
assert the *SQL semantics* (ON CONFLICT ...) without needing a database."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from src.pipeline.models import Chunk, ProcessedPaper
from src.storage.repositories import PgRepo


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


@pytest.fixture
def processed(make_paper):
    paper = make_paper("2401.00001v1")
    chunks = [
        Chunk(chunk_id=f"c{i}", paper_id=paper.arxiv_id, section_title="Intro", content=f"text {i}")
        for i in range(3)
    ]
    return ProcessedPaper(metadata=paper, sections=[], chunks=chunks)


def _session(existing: bool):
    session = AsyncMock()
    lookup = MagicMock()
    lookup.scalars.return_value.first.return_value = object() if existing else None
    session.execute.return_value = lookup
    return session


async def test_existing_paper_is_skipped_without_writing(processed):
    session = _session(existing=True)
    assert await PgRepo().upsert_paper_and_chunks(session, processed) is False
    assert session.execute.await_count == 1  # only the existence check
    session.commit.assert_not_awaited()


async def test_new_paper_inserts_paper_and_chunks_and_commits_once(processed):
    session = _session(existing=False)
    assert await PgRepo().upsert_paper_and_chunks(session, processed) is True
    assert session.execute.await_count == 3  # select, insert paper, insert chunks
    session.commit.assert_awaited_once()


async def test_inserts_are_conflict_safe(processed):
    session = _session(existing=False)
    await PgRepo().upsert_paper_and_chunks(session, processed)
    paper_sql = _sql(session.execute.await_args_list[1].args[0])
    chunk_sql = _sql(session.execute.await_args_list[2].args[0])
    assert "ON CONFLICT (arxiv_id) DO NOTHING" in paper_sql
    assert "ON CONFLICT (chunk_id) DO NOTHING" in chunk_sql


async def test_chunk_insert_failure_never_commits_the_paper(processed):
    """Half a paper (row but no chunks) must not become visible."""
    session = _session(existing=False)
    lookup = session.execute.return_value
    session.execute.side_effect = [lookup, MagicMock(), RuntimeError("chunk insert blew up")]
    with pytest.raises(RuntimeError):
        await PgRepo().upsert_paper_and_chunks(session, processed)
    session.commit.assert_not_awaited()


async def test_dlq_write_is_an_upsert_that_increments_retry_count():
    session = AsyncMock()
    await PgRepo().log_failure_to_dlq(session, "2401.00001v1", "boom")
    sql = _sql(session.execute.await_args.args[0])
    assert "ON CONFLICT (arxiv_id) DO UPDATE" in sql
    assert "retry_count" in sql
    session.commit.assert_awaited_once()


async def test_chunk_metadata_is_written_to_the_metadata_column(processed):
    session = _session(existing=False)
    await PgRepo().upsert_paper_and_chunks(session, processed)
    chunk_sql = _sql(session.execute.await_args_list[2].args[0])
    assert "(chunk_id, paper_id, section_title, content, metadata)" in chunk_sql


# ---------- equations, references, parse provenance


@pytest.fixture
def processed_with_math(make_paper):
    from src.pipeline.models import Equation, Reference

    paper = make_paper("2401.00002v1")
    chunks = [
        Chunk(chunk_id=f"c{i}", paper_id=paper.arxiv_id, section_title="S", content=f"t {i}")
        for i in range(2)
    ]
    return ProcessedPaper(
        metadata=paper,
        sections=[],
        chunks=chunks,
        equations=[Equation(ordinal=0, latex=r"\frac{a}{b}", number="(1)", section_path="1 Intro")],
        references=[Reference(ordinal=0, key="b1", text="A. Author. Title.", urls=["https://x.y"])],
        parse_method="html",
        math_fidelity="latex",
        parse_stats={"words": 10},
    )


async def test_equations_and_references_are_written_in_the_same_transaction(processed_with_math):
    session = _session(existing=False)
    assert await PgRepo().upsert_paper_and_chunks(session, processed_with_math) is True
    assert session.execute.await_count == 5  # select, paper, chunks, equations, references
    assert "paper_equations" in _sql(session.execute.await_args_list[3].args[0])
    assert "paper_references" in _sql(session.execute.await_args_list[4].args[0])
    session.commit.assert_awaited_once()  # ONE commit: all of it or none of it


async def test_paper_row_records_parse_method_and_math_fidelity(processed_with_math):
    session = _session(existing=False)
    await PgRepo().upsert_paper_and_chunks(session, processed_with_math)
    paper_sql = _sql(session.execute.await_args_list[1].args[0])
    assert "parse_method" in paper_sql and "math_fidelity" in paper_sql


async def test_a_failing_equation_insert_never_commits_the_paper(processed_with_math):
    session = _session(existing=False)
    lookup = session.execute.return_value
    session.execute.side_effect = [lookup, MagicMock(), MagicMock(), RuntimeError("bad equation")]
    with pytest.raises(RuntimeError):
        await PgRepo().upsert_paper_and_chunks(session, processed_with_math)
    session.commit.assert_not_awaited()


async def test_delete_paper_removes_the_row_and_commits():
    session = AsyncMock()
    await PgRepo().delete_paper(session, "2401.00001v1")
    assert _sql(session.execute.await_args.args[0]).startswith("DELETE FROM papers")
    session.commit.assert_awaited_once()
