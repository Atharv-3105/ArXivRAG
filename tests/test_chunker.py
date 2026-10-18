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


# ---------- math-aware chunking, section context


def _math_spans(text):
    import re

    return re.findall(r"\$\$.+?\$\$|\$.+?\$", text, flags=re.DOTALL)


def test_a_formula_is_never_split_across_chunks(chunker, make_section):
    """Half a formula is worse than useless. Formulas with internal spaces are the risky case."""
    formulas = [f"$x_{{{i}}} + y_{{{i}}} = \\frac{{a}}{{b}} \\, z_{{{i}}}$" for i in range(60)]
    words_between = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    text = " ".join(f"{words_between} {f}" for f in formulas)
    chunks = chunker.chunk_sections("p", [make_section(text)])
    assert len(chunks) > 3
    for c in chunks:
        assert c.content.count("$") % 2 == 0, c.content  # balanced delimiters in every chunk
    seen = {m for c in chunks for m in _math_spans(c.content)}
    assert set(formulas) <= seen  # every formula survives intact in at least one chunk


def test_display_math_with_a_tag_stays_in_one_piece(chunker, make_section):
    eq = r"$$\sum_{i=1}^{n} a_i b_i \;\;=\;\; \int_0^1 f(x)\,dx \tag{7}$$"
    text = words(140) + " " + eq + " " + words(140, "tail")
    chunks = chunker.chunk_sections("p", [make_section(text)])
    holders = [c for c in chunks if "\\tag{7}" in c.content]
    assert holders and all(eq in c.content for c in holders)


def test_an_oversized_formula_becomes_a_big_chunk_not_a_broken_one(make_section):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    huge = "$$" + " + ".join(f"a_{{{i}}}" for i in range(120)) + "$$"  # far longer than 200 chars
    text = words(60) + " " + huge + " " + words(60, "tail")
    chunks = chunker.chunk_sections("p", [make_section(text)])
    assert any(huge in c.content for c in chunks)
    assert all(c.content.count("$") % 2 == 0 for c in chunks)


def test_chunks_carry_their_section_path_as_context(chunker, make_section):
    section = make_section(words(300), title="Method")
    section.path = "3 Method > 3.1 Setup"
    chunks = chunker.chunk_sections("p", [section])
    assert all(c.context == "3 Method > 3.1 Setup" for c in chunks)
    assert chunks[0].embedding_text.startswith("3 Method > 3.1 Setup\n")
    assert chunks[0].metadata["section_path"] == "3 Method > 3.1 Setup"


def test_context_falls_back_to_the_title_when_there_is_no_path(chunker, make_section):
    chunks = chunker.chunk_sections("p", [make_section(words(300), title="Results")])
    assert chunks[0].context == "Results"


def test_math_span_count_and_anchor_are_recorded(chunker, make_section):
    section = make_section("intro " + "text $a$ and $b$ and $$c$$ " * 10 + words(50))
    section.anchor = "S2.SS1"
    [first, *_] = chunker.chunk_sections("p", [section])
    assert first.metadata["math_spans"] >= 3
    assert first.metadata["anchor"] == "S2.SS1"


def test_overlong_titles_are_truncated_to_fit_the_column(chunker, make_section):
    chunks = chunker.chunk_sections("p", [make_section(words(300), title="T" * 400)])
    assert len(chunks[0].section_title) == 255


def test_escaped_dollars_are_prose_not_math(chunker, make_section):
    """A price like \$100 must neither be counted as math nor pair with a later formula."""
    text = (
        "The API costs \$100 per run and \$5 per query. "
        + words(60)
        + " so $x + y$ holds. "
        + words(60, "z")
    )
    chunks = chunker.chunk_sections("p", [make_section(text)])
    joined = " ".join(c.content for c in chunks)
    assert "\$100" in joined and "$x + y$" in joined
    assert sum(c.metadata["math_spans"] for c in chunks) == 1  # only the real formula


def test_a_formula_containing_an_escaped_dollar_stays_whole(chunker, make_section):
    formula = r"$\text{cost} = \$5 + x$"
    text = words(120) + " " + formula + " " + words(120, "tail")
    chunks = chunker.chunk_sections("p", [make_section(text)])
    assert any(formula in c.content for c in chunks)


# ------------------------------------------------------------------ token-budget mode


def wordpiece_like(word: str) -> int:
    """A stand-in for WordPiece: ~3 characters per token, and maths is far more expensive, which
    is exactly why a character budget cannot protect a model's window."""
    return max(1, len(word) // 3) + (2 * word.count("\\") if "$" in word else 0)


@pytest.fixture
def token_chunker():
    return TextChunker(token_counter=wordpiece_like, max_tokens=60, overlap_tokens=10)


def _tokens(chunk_text: str) -> int:
    return sum(wordpiece_like(w) for w in chunk_text.split(" "))


def test_token_mode_never_overflows_the_budget(token_chunker, make_section):
    text = " ".join(
        w if i % 7 else r"$\alpha_{i}\beta\gamma\delta$" for i, w in enumerate(words(600).split())
    )
    chunks = token_chunker.chunk_sections("p", [make_section(text)])
    assert len(chunks) > 3
    for c in chunks:
        assert _tokens(c.content) <= 60, c.content


def test_token_mode_beats_a_character_budget_on_math_heavy_text(make_section):
    """The measured failure: the same characters, many more tokens when the text is LaTeX."""
    math_heavy = " ".join([r"$\mathbf{x}\in\mathbb{R}^{d}$"] * 120)
    by_chars = TextChunker(chunk_size=300, chunk_overlap=50).chunk_sections(
        "p", [make_section(math_heavy)]
    )
    by_tokens = TextChunker(
        token_counter=wordpiece_like, max_tokens=60, overlap_tokens=10
    ).chunk_sections("p", [make_section(math_heavy)])
    assert max(_tokens(c.content) for c in by_chars) > 60  # would be silently truncated
    assert max(_tokens(c.content) for c in by_tokens) <= 60


def test_token_mode_loses_no_text_and_overlaps(token_chunker, make_section):
    text = words(400)
    chunks = token_chunker.chunk_sections("p", [make_section(text)])
    joined = chunks[0].content.split()
    for nxt in chunks[1:]:
        words_next = nxt.content.split()
        assert words_next[0] in joined[-12:]  # the tail of the previous chunk is carried over
        joined.extend(words_next)
    assert set(text.split()) <= set(joined)
    assert chunks[-1].content.split()[-1] == text.split()[-1]


def test_token_mode_keeps_a_formula_whole_even_when_it_exceeds_the_budget(
    token_chunker, make_section
):
    huge = "$" + r"\sum_{i}" * 40 + "$"  # far over 60 tokens on its own
    text = f"{words(40)} {huge} {words(40)}"
    chunks = token_chunker.chunk_sections("p", [make_section(text)])
    assert sum(huge in c.content for c in chunks) >= 1
    assert all(c.content.count("$") % 2 == 0 for c in chunks)


def test_token_mode_rejects_an_overlap_that_is_not_smaller_than_the_budget():
    with pytest.raises(ValueError):
        TextChunker(token_counter=wordpiece_like, max_tokens=20, overlap_tokens=20)


def test_the_section_path_spends_part_of_the_token_budget(make_section):
    """The path is embedded with every chunk, so path + content must fit, not just the content."""
    long_path = "Appendix > Details of the very long experimental protocol for the ablation"
    section = make_section(words(500))
    section.path = long_path
    chunker = TextChunker(token_counter=wordpiece_like, max_tokens=60, overlap_tokens=10)
    reserved = sum(wordpiece_like(w) for w in long_path.split()) + 2
    for c in chunker.chunk_sections("p", [section]):
        assert _tokens(c.content) + reserved <= 60, c.content
