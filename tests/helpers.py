"""Shared test builders. Import as `from tests.helpers import ...` (pythonpath is the repo root)."""

from pathlib import Path

import pymupdf


def words(n: int, prefix: str = "word") -> str:
    """n distinct, deterministic words (distinct so overlap/coverage can be asserted)."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def make_pdf(path: Path, lines: list[str], lines_per_page: int = 60) -> Path:
    """Write a real PDF, one text line per call so pymupdf extracts them in order.

    We deliberately build real PDFs (not mocks) so the parser is tested against the same
    library behaviour it sees in production.
    """
    doc = pymupdf.open()
    for start in range(0, max(len(lines), 1), lines_per_page):
        page = doc.new_page()
        y = 50.0
        for line in lines[start : start + lines_per_page]:
            page.insert_text((50, y), line, fontsize=9)
            y += 11
    doc.save(str(path))
    doc.close()
    return path


def body_lines(n_words: int, words_per_line: int = 12, prefix: str = "body") -> list[str]:
    """Body text that can never look like a section header."""
    tokens = [f"{prefix}{i}" for i in range(n_words)]
    return [" ".join(tokens[i : i + words_per_line]) for i in range(0, len(tokens), words_per_line)]


def atom_entry(
    arxiv_id: str = "2401.00001v1",
    title: str = "A Paper",
    *,
    with_title: bool = True,
    with_pdf: bool = True,
    categories: tuple[str, ...] = ("cs.LG", "cs.AI"),
) -> str:
    title_xml = f"<title>{title}</title>" if with_title else ""
    pdf_xml = (
        f'<link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}" rel="related" '
        f'type="application/pdf"/>'
        if with_pdf
        else ""
    )
    cats = "".join(f'<category term="{c}"/>' for c in categories)
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <updated>2024-01-02T10:00:00Z</updated>
    <published>2024-01-01T10:00:00Z</published>
    {title_xml}
    <summary> An abstract. </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    {pdf_xml}
    {cats}
  </entry>"""


def atom_feed(*entries: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">' + "".join(entries) + "\n</feed>"
    )
