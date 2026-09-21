"""PaperExtractor: tier selection (HTML first, PDF fallback) and caching."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.exceptions import FetchError, ParseError
from src.pipeline.extractor import PaperExtractor
from src.pipeline.html_parser import HtmlParser
from src.pipeline.models import SectionContent
from src.pipeline.parser import ParseMetrics
from tests.test_html_parser import FILLER, page, para, section

GOOD_HTML = page(section("S1", "1", "Introduction", para(FILLER)))


class Env:
    """A fake fetcher + fake PDF parser, wired to the real HtmlParser (small word threshold)."""

    def __init__(self, tmp_path):
        self.html_response: str | None = GOOD_HTML
        self.html_error: Exception | None = None
        self.fetcher = MagicMock()
        self.fetcher.fetch_html = AsyncMock(side_effect=self._fetch_html)
        self.fetcher.download_pdf = AsyncMock(side_effect=self._download)
        self.pdf_parser = MagicMock()
        self.pdf_parser.parse = MagicMock(side_effect=self._parse_pdf)
        self.pdf_error: Exception | None = None
        self.extractor = PaperExtractor(
            self.fetcher,
            tmp_path,
            html_parser=HtmlParser(min_words=5),
            pdf_parser=self.pdf_parser,
        )

    async def _fetch_html(self, paper):
        if self.html_error:
            raise self.html_error
        return self.html_response

    async def _download(self, paper, path):
        with open(path, "wb") as f:
            f.write(b"%PDF fake")
        return True

    def _parse_pdf(self, path):
        if self.pdf_error:
            raise self.pdf_error
        section_ = SectionContent(title="Intro", text="pdf text", page_start=1, page_end=1)
        return [section_], ParseMetrics()


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# ---------------------------------------------------------------- tier 1


async def test_html_is_preferred_and_no_pdf_is_downloaded(env, make_paper):
    doc = await env.extractor.extract(make_paper())
    assert (doc.method, doc.math_fidelity) == ("html", "latex")
    env.fetcher.download_pdf.assert_not_awaited()
    env.pdf_parser.parse.assert_not_called()


async def test_html_is_cached_so_a_second_extract_does_not_refetch(env, make_paper):
    paper = make_paper()
    await env.extractor.extract(paper)
    await env.extractor.extract(paper)
    assert env.fetcher.fetch_html.await_count == 1


# ---------------------------------------------------------------- tier 2


async def test_papers_without_html_fall_back_to_the_pdf(env, make_paper):
    env.html_response = None  # arXiv answered 404
    doc = await env.extractor.extract(make_paper())
    assert (doc.method, doc.math_fidelity) == ("pdf", "degraded")
    assert doc.stats["html_fallback_reason"] == "no_html"
    env.fetcher.download_pdf.assert_awaited_once()


async def test_a_rejected_html_parse_falls_back_and_records_why(env, make_paper):
    env.html_response = "<html><body>not a LaTeXML document</body></html>"
    doc = await env.extractor.extract(make_paper())
    assert doc.method == "pdf"
    assert doc.stats["html_fallback_reason"] == "PARSE_HTML_ERROR"


async def test_html_fetch_outage_does_not_block_ingestion(env, make_paper):
    """HTML is an optimisation. If arXiv's HTML host is down, the PDF tier still ingests."""
    env.html_error = FetchError("503", code="FETCH_HTML_ERROR")
    doc = await env.extractor.extract(make_paper())
    assert doc.method == "pdf"
    assert doc.stats["html_fallback_reason"] == "FETCH_HTML_ERROR"


async def test_existing_pdf_is_not_downloaded_again(env, make_paper, tmp_path):
    env.html_response = None
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / "2401.00001v1.pdf").write_bytes(b"%PDF cached")
    await env.extractor.extract(make_paper("2401.00001v1"))
    env.fetcher.download_pdf.assert_not_awaited()


async def test_pdf_failure_propagates_when_no_tier_can_parse(env, make_paper):
    env.html_response = None
    env.pdf_error = ParseError("too short", code="PARSE_LENGTH_ERROR")
    with pytest.raises(ParseError):
        await env.extractor.extract(make_paper())


async def test_pdf_download_failure_propagates(env, make_paper):
    env.html_response = None
    env.fetcher.download_pdf = AsyncMock(side_effect=FetchError("404", code="FETCH_PDF_ERROR"))
    with pytest.raises(FetchError):
        await env.extractor.extract(make_paper())


# --------------------------------------------------------------- filenames


async def test_old_style_arxiv_ids_are_safe_as_filenames(env, make_paper, tmp_path):
    env.html_response = None
    await env.extractor.extract(make_paper("hep-th/9901001v1"))
    assert (tmp_path / "pdf" / "hep-th_9901001v1.pdf").exists()


async def test_cache_lives_under_html_and_pdf_directories(env, make_paper, tmp_path):
    await env.extractor.extract(make_paper("2401.00001v1"))
    assert (tmp_path / "html" / "2401.00001v1.html").exists()
    assert not list((tmp_path / "html").glob("*.part"))  # atomic write leaves no temp file
