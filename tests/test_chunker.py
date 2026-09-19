"""TextChunker: pure logic, no I/O. Each test states the property it protects."""

import pytest

from src.pipeline.chunker import TextChunker
from tests.helpers import words

CHUNK_SIZE = 1000
OVERLAP = 200


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)


def test_no_sections_yields_no_chunks(chunker):
    assert chunker.chunk_sections("2401.00001v1", []) == []


def test_very_short_sections_are_skipped(chunker, make_section):
    assert chunker.chunk_sections("p", [make_section("too short")]) == []


def test_section_shorter_than_chunk_size_is_one_chunk(chunker, make_section):
    text = words(50)  # ~400 chars
    chunks = chunker.chunk_sections("p", [make_section(text)])
    assert len(chunks) == 1
    assert chunks[0].content == text


def test_chunks_respect_chunk_size(chunker, make_section):
    chunks = chunker.chunk_sections("p", [make_section(words(3000))])
    longest_token = len("word2999")
    assert chunks, "expected chunks"
    for c in chunks:
        assert len(c.content) <= CHUNK_SIZE + longest_token + 1


def test_chunk_count_is_proportional_to_text_length(chunker, make_section):
    """3000 words is ~27k chars; with 1000-char chunks that is tens of chunks, not thousands."""
    chunks = chunker.chunk_sections("p", [make_section(words(3000))])
    assert 10 < len(chunks) < 100


def test_consecutive_chunks_overlap(chunker, make_section):
    chunks = chunker.chunk_sections("p", [make_section(words(3000))])
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert set(a.content.split()) & set(b.content.split()), "no shared context between chunks"


def test_no_text_is_lost(chunker, make_section):
    text = words(3000)
    chunks = chunker.chunk_sections("p", [make_section(text)])
    covered = {w for c in chunks for w in c.content.split()}
    assert covered == set(text.split())


def test_chunk_ids_are_unique_within_a_paper(chunker, make_section):
    chunks = chunker.chunk_sections("p", [make_section(words(3000))])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_unique_when_sections_share_a_title(chunker, make_section):
    """Real papers repeat titles (e.g. two 'Experiments' blocks after a bad split)."""
    sections = [
        make_section(words(300, "a"), title="Experiments"),
        make_section(words(300, "b"), title="Experiments"),
    ]
    ids = [c.chunk_id for c in chunker.chunk_sections("p", sections)]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_deterministic(chunker, make_section):
    """Idempotency: re-ingesting the same paper must produce the same ids (upsert, not dupes)."""
    sections = [make_section(words(500))]
    first = [c.chunk_id for c in chunker.chunk_sections("p", sections)]
    second = [c.chunk_id for c in chunker.chunk_sections("p", sections)]
    assert first == second


def test_chunks_carry_paper_and_section_provenance(chunker, make_section):
    chunks = chunker.chunk_sections("2401.00001v1", [make_section(words(500), title="Methods")])
    assert all(c.paper_id == "2401.00001v1" for c in chunks)
    assert all(c.section_title == "Methods" for c in chunks)


def test_word_count_metadata_matches_content(chunker, make_section):
    for c in chunker.chunk_sections("p", [make_section(words(500))]):
        assert c.metadata["word_count"] == len(c.content.split())


def test_section_without_title_still_chunks(chunker, make_section):
    """SectionContent.title is Optional; Chunk.section_title is required str."""
    section = make_section(words(300))
    section.title = None
    chunks = chunker.chunk_sections("p", [section])
    assert chunks
    assert all(isinstance(c.section_title, str) for c in chunks)
