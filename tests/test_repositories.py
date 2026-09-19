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
