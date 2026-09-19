"""ArxivFetcher against a mocked arXiv (httpx.MockTransport). No network."""

import asyncio

import httpx
import pytest

from src.core.exceptions import FetchError
from src.pipeline.fetcher import ArxivFetcher
from tests.helpers import atom_entry, atom_feed


class _Calls:
    """Records requests and replays a scripted list of responses (last one repeats)."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        idx = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[idx]


def _ok(body: str) -> httpx.Response:
    return httpx.Response(200, text=body)


@pytest.fixture
async def make_fetcher(monkeypatch):
    """Build a fetcher wired to a scripted transport.

    asyncio.sleep is stubbed so rate-limit delays and retry backoff cost nothing. Patching the sleep
    (rather than a specific retry decorator) keeps these tests independent of *how* retries are
    implemented; they only assert *that* they happen.
    """
    created: list[ArxivFetcher] = []

    async def _instant(_seconds: float = 0) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)

    def _build(*responses: httpx.Response) -> tuple[ArxivFetcher, _Calls]:
        calls = _Calls(*responses)
        fetcher = ArxivFetcher()
        fetcher._client = httpx.AsyncClient(transport=httpx.MockTransport(calls))
        created.append(fetcher)
        return fetcher, calls

    yield _build
    for f in created:
        await f._client.aclose()


# ---------- metadata fetching ----------


async def test_parses_entries_into_metadata(make_fetcher):
    fetcher, _ = make_fetcher(
        _ok(atom_feed(atom_entry("2401.00001v1", "First"), atom_entry("2401.00002v1")))
    )
    papers = await fetcher.fetch_papers("all:transformers")

    assert [p.arxiv_id for p in papers] == ["2401.00001v1", "2401.00002v1"]
    first = papers[0]
    assert first.title == "First"
    assert first.authors == ["Ada Lovelace", "Alan Turing"]
    assert first.categories == ["cs.LG", "cs.AI"]
    assert first.primary_category == "cs.LG"
    assert first.pdf_url == "http://arxiv.org/pdf/2401.00001v1"
    assert first.published.tzinfo is not None


async def test_sends_query_parameters(make_fetcher):
    fetcher, calls = make_fetcher(_ok(atom_feed()))
    await fetcher.fetch_papers("all:rag", max_results=7)
    params = calls.requests[0].url.params
    assert params["search_query"] == "all:rag"
    assert params["max_results"] == "7"
    assert params["sortBy"] == "submittedDate"


async def test_empty_feed_returns_empty_list(make_fetcher):
    fetcher, _ = make_fetcher(_ok(atom_feed()))
    assert await fetcher.fetch_papers("all:nothing") == []


async def test_malformed_entry_is_skipped_not_fatal(make_fetcher):
    feed = atom_feed(atom_entry("2401.00001v1", with_title=False), atom_entry("2401.00002v1"))
    fetcher, _ = make_fetcher(_ok(feed))
    papers = await fetcher.fetch_papers("all:x")
    assert [p.arxiv_id for p in papers] == ["2401.00002v1"]


async def test_entry_without_pdf_link_is_skipped_not_fatal(make_fetcher):
    feed = atom_feed(atom_entry("2401.00001v1", with_pdf=False), atom_entry("2401.00002v1"))
    fetcher, _ = make_fetcher(_ok(feed))
    papers = await fetcher.fetch_papers("all:x")
    assert [p.arxiv_id for p in papers] == ["2401.00002v1"]


async def test_old_style_arxiv_id_keeps_its_archive_prefix(make_fetcher):
    fetcher, _ = make_fetcher(_ok(atom_feed(atom_entry("hep-th/9901001v1"))))
    papers = await fetcher.fetch_papers("all:x")
    assert papers[0].arxiv_id == "hep-th/9901001v1"


async def test_paging_offset_is_sent(make_fetcher):
    fetcher, calls = make_fetcher(_ok(atom_feed()))
    await fetcher.fetch_papers("all:x", max_results=10, start=20)
    assert calls.requests[0].url.params["start"] == "20"


async def test_client_error_is_not_retried(make_fetcher):
    """A 400 (bad query) will never succeed; retrying just hammers arXiv."""
    fetcher, calls = make_fetcher(httpx.Response(400))
    with pytest.raises(FetchError):
        await fetcher.fetch_papers("all:x")
    assert len(calls.requests) == 1


async def test_html_error_page_is_not_saved_as_a_pdf(make_fetcher, make_paper, tmp_path):
    fetcher, _ = make_fetcher(httpx.Response(200, content=b"<html>rate limited</html>"))
    target = tmp_path / "p.pdf"
    with pytest.raises(FetchError):
        await fetcher.download_pdf(make_paper(), str(target))
    assert not target.exists()


# ---------- resilience (plan: retry with backoff+jitter, bounded, typed errors) ----------


async def test_transient_server_error_is_retried(make_fetcher):
    fetcher, calls = make_fetcher(
        httpx.Response(503),
        httpx.Response(503),
        _ok(atom_feed(atom_entry("2401.00001v1"))),
    )
    papers = await fetcher.fetch_papers("all:x")
    assert len(papers) == 1
    assert len(calls.requests) == 3


async def test_persistent_failure_raises_fetch_error_after_bounded_retries(make_fetcher):
    fetcher, calls = make_fetcher(httpx.Response(503))
    with pytest.raises(FetchError) as exc:
        await fetcher.fetch_papers("all:x")
    assert exc.value.code == "FETCH_API_ERROR"
    assert 1 < len(calls.requests) <= 3, "must retry, but never forever"


async def test_malformed_xml_raises_fetch_error(make_fetcher):
    """A 200 with garbage body is still an arXiv failure; the runner only handles FetchError."""
    fetcher, _ = make_fetcher(_ok("<feed><entry>"))
    with pytest.raises(FetchError):
        await fetcher.fetch_papers("all:x")


# ---------- PDF download ----------


class _DiesMidway(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"%PDF-1.4 partial-bytes"
        raise httpx.ReadError("connection reset by peer")


async def test_download_pdf_writes_file(make_fetcher, make_paper, tmp_path):
    fetcher, _ = make_fetcher(httpx.Response(200, content=b"%PDF-1.4 fake"))
    target = tmp_path / "p.pdf"
    assert await fetcher.download_pdf(make_paper(), str(target)) is True
    assert target.read_bytes() == b"%PDF-1.4 fake"


async def test_download_http_error_raises_fetch_error(make_fetcher, make_paper, tmp_path):
    fetcher, _ = make_fetcher(httpx.Response(404))
    with pytest.raises(FetchError) as exc:
        await fetcher.download_pdf(make_paper(), str(tmp_path / "p.pdf"))
    assert exc.value.code == "FETCH_PDF_ERROR"


async def test_interrupted_download_leaves_no_partial_file(make_fetcher, make_paper, tmp_path):
    """Runner skips download if the file exists, so a partial file = permanently corrupt paper."""
    fetcher, _ = make_fetcher(httpx.Response(200, stream=_DiesMidway()))
    target = tmp_path / "p.pdf"
    with pytest.raises(FetchError):
        await fetcher.download_pdf(make_paper(), str(target))
    assert not target.exists()


async def test_transient_download_error_is_retried(make_fetcher, make_paper, tmp_path):
    fetcher, calls = make_fetcher(httpx.Response(503), httpx.Response(200, content=b"%PDF-1.4 ok"))
    target = tmp_path / "p.pdf"
    assert await fetcher.download_pdf(make_paper(), str(target)) is True
    assert target.read_bytes() == b"%PDF-1.4 ok"
    assert len(calls.requests) == 2
