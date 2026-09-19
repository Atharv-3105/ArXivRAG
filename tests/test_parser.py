"""PDFParser against real PDFs generated on the fly (see tests/helpers.make_pdf)."""

import pytest

from src.core.exceptions import ParseError
from src.pipeline.parser import PDFParser
from tests.helpers import body_lines, make_pdf


@pytest.fixture
def parser():
    return PDFParser()


def _structured_paper(path):
    lines = (
        ["Abstract"]
        + body_lines(200, prefix="abs")
        + ["1. Introduction"]
        + body_lines(300, prefix="intro")
        + ["2. Methods"]
        + body_lines(300, prefix="meth")
        + ["MARKER-SENTENCE-IN-METHODS end of methods"]
        + ["3. Results"]
        + body_lines(300, prefix="res")
    )
    return make_pdf(path, lines)


def test_splits_numbered_sections(parser, tmp_path):
    sections, metrics = parser.parse(str(_structured_paper(tmp_path / "p.pdf")))
    titles = [s.title for s in sections]
    assert {"Introduction", "Methods", "Results"} <= set(titles)
    assert metrics.is_valid
    assert metrics.section_count == len(sections)


def test_section_body_text_is_kept_under_its_own_section(parser, tmp_path):
    sections, _ = parser.parse(str(_structured_paper(tmp_path / "p.pdf")))
    methods = next(s for s in sections if s.title == "Methods")
    assert "meth10" in methods.text
    assert "MARKER-SENTENCE-IN-METHODS" in methods.text
    assert "intro10" not in methods.text  # not bleeding across section boundaries


def test_no_words_are_dropped_by_section_splitting(parser, tmp_path):
    sections, metrics = parser.parse(str(_structured_paper(tmp_path / "p.pdf")))
    kept = sum(len(s.text.split()) for s in sections)
    # Headers are allowed to be re-labelled, so allow a small tolerance.
    assert kept >= metrics.total_words * 0.95


def test_sections_report_a_valid_page_range(parser, tmp_path):
    sections, metrics = parser.parse(str(_structured_paper(tmp_path / "p.pdf")))
    for s in sections:
        assert 1 <= s.page_start <= s.page_end <= metrics.total_pages


def test_too_short_paper_is_rejected(parser, tmp_path):
    pdf = make_pdf(tmp_path / "short.pdf", ["1. Intro", "a few words", "2. End", "more words"])
    with pytest.raises(ParseError) as exc:
        parser.parse(str(pdf))
    assert exc.value.code == "PARSE_LENGTH_ERROR"


def test_paper_without_detectable_structure_is_rejected(parser, tmp_path):
    pdf = make_pdf(tmp_path / "flat.pdf", body_lines(1500))
    with pytest.raises(ParseError) as exc:
        parser.parse(str(pdf))
    assert exc.value.code == "PARSE_STRUCTURE_ERROR"


def test_latex_blocks_are_counted_once_not_per_page(parser, tmp_path):
    """Two \\begin{...} blocks in the paper must count as 2 regardless of page count."""
    lines = (
        ["1. Introduction"]
        + body_lines(200, prefix="a")
        + [r"\begin{equation} x = y \end{equation}", r"\begin{align} a = b \end{align}"]
        + body_lines(400, prefix="b")
        + ["2. Methods"]
        + body_lines(400, prefix="c")
    )
    pdf = make_pdf(tmp_path / "latex.pdf", lines, lines_per_page=25)  # forces many pages
    _, metrics = parser.parse(str(pdf))
    assert metrics.total_pages > 3
    assert metrics.latex_block_count == 2


def test_corrupt_pdf_raises_parse_error_not_a_library_error(parser, tmp_path):
    """Runner routes ParseError to the DLQ with a code; raw library errors become UNKNOWN."""
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    with pytest.raises(ParseError):
        parser.parse(str(bad))


def _titles(parser, path):
    sections, _ = parser.parse(str(path))
    return [s.title for s in sections]


def test_number_and_title_on_separate_lines(parser, tmp_path):
    """How many real arXiv PDFs lay out headings: '1' then 'Introduction' on the next line."""
    lines = (
        ["1", "Introduction"]
        + body_lines(300, prefix="a")
        + ["2", "Related Work"]
        + body_lines(300, prefix="b")
        + ["3", "Method"]
        + body_lines(300, prefix="c")
    )
    titles = _titles(parser, make_pdf(tmp_path / "p.pdf", lines))
    assert titles == ["Introduction", "Related Work", "Method"]


def test_unnumbered_uppercase_headings(parser, tmp_path):
    lines = (
        ["INTRODUCTION"]
        + body_lines(300, prefix="a")
        + ["METHODS"]
        + body_lines(300, prefix="b")
        + ["CONCLUSION"]
        + body_lines(300, prefix="c")
    )
    titles = _titles(parser, make_pdf(tmp_path / "p.pdf", lines))
    assert titles == ["Introduction", "Methods", "Conclusion"]


def test_a_numbered_sentence_is_not_a_heading(parser, tmp_path):
    lines = (
        ["1. Introduction"]
        + body_lines(300, prefix="a")
        + ["2. Models that are generally used in practice today"]
        + body_lines(300, prefix="b")
        + ["3. Methods"]
        + body_lines(300, prefix="c")
    )
    titles = _titles(parser, make_pdf(tmp_path / "p.pdf", lines))
    assert titles == ["Introduction", "Methods"]


def test_reference_list_is_dropped(parser, tmp_path):
    lines = (
        ["1. Introduction"]
        + body_lines(300, prefix="a")
        + ["2. Methods"]
        + body_lines(300, prefix="b")
        + ["References"]
        + body_lines(200, prefix="ref")
    )
    sections, _ = parser.parse(str(make_pdf(tmp_path / "p.pdf", lines)))
    assert "References" not in [s.title for s in sections]
    assert not any("ref10" in s.text for s in sections)


def test_control_characters_are_stripped_from_extracted_text():
    """Real PDFs yield NUL bytes; Postgres TEXT rejects them, which sank 2 of 5 papers."""
    from src.pipeline.parser import _clean_text

    assert _clean_text("a\x00b\x01c\x1fd") == "abcd"
    assert _clean_text("keep\nnewlines\tand tabs") == "keep\nnewlines\tand tabs"


def test_parsed_sections_never_contain_nul_bytes(parser, tmp_path, monkeypatch):
    import pymupdf

    real_get_text = pymupdf.Page.get_text
    monkeypatch.setattr(
        pymupdf.Page, "get_text", lambda self, *a, **k: real_get_text(self, *a, **k) + "\x00"
    )
    sections, _ = parser.parse(str(_structured_paper(tmp_path / "p.pdf")))
    assert all("\x00" not in s.text for s in sections)
