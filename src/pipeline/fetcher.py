import asyncio
import contextlib
import os
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import structlog
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from src.core.config import settings
from src.core.exceptions import FetchError
from src.pipeline.models import PaperMetadata

logger = structlog.get_logger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _is_transient(exc: Exception) -> bool:
    """Worth retrying: timeouts / connection problems / 429 / 5xx. A 404 or bad query is not."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)


def _should_retry(exc: BaseException) -> bool:
    return isinstance(exc, FetchError) and bool(exc.metadata.get("retryable"))


# Retry FetchErrors that were flagged transient, with exponential backoff + jitter.
_retry_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1.0, max=10.0),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)


class ArxivFetcher:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self):
        self._semaphore = asyncio.Semaphore(1)  # arXiv asks for strictly sequential requests
        self._client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "ArXivMind/0.1 (research paper ingestion)"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @asynccontextmanager
    async def _throttled(self) -> AsyncIterator[None]:
        """One request at a time, with a pause after each (also after failures)."""
        async with self._semaphore:
            try:
                yield
            finally:
                await asyncio.sleep(settings.ARXIV_REQUEST_DELAY_SECONDS)

    @_retry_transient
    async def fetch_papers(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str | None = None,
        sort_order: str = "descending",
    ) -> list[PaperMetadata]:
        """Fetch paper metadata from the arXiv API (rate-limited, retried on transient errors)."""
        params: dict[str, str | int] = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by or settings.ARXIV_SORT_BY,
            "sortOrder": sort_order,
        }
        logger.info("Fetching papers from ArXiv", query=query, max_results=max_results, start=start)

        async with self._throttled():
            try:
                response = await self._client.get(self.BASE_URL, params=params)
                response.raise_for_status()
            except Exception as e:
                raise FetchError(
                    f"ArXiv API request failed: {e}",
                    code="FETCH_API_ERROR",
                    metadata={"query": query, "retryable": _is_transient(e)},
                ) from e

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as e:
            raise FetchError(
                f"ArXiv returned malformed XML: {e}",
                code="FETCH_API_ERROR",
                metadata={"query": query, "retryable": False},
            ) from e

        papers: list[PaperMetadata] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            try:
                papers.append(self._parse_entry(entry))
            except Exception as e:
                logger.warning("Failed to parse ArXiv entry", error=str(e))

        logger.info("Parsed papers", count=len(papers))
        return papers

    def _parse_entry(self, entry: ET.Element) -> PaperMetadata:
        def text(tag: str) -> str:
            node = entry.find(f"atom:{tag}", ATOM_NS)
            if node is None or node.text is None:
                raise ValueError(f"entry has no <{tag}>")
            return node.text

        authors = [
            name.text
            for a in entry.findall("atom:author", ATOM_NS)
            if (name := a.find("atom:name", ATOM_NS)) is not None and name.text
        ]
        categories = [c.get("term", "") for c in entry.findall("atom:category", ATOM_NS)]

        pdf_link = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf":
                pdf_link = link.get("href")

        # "http://arxiv.org/abs/2401.12345v1" -> "2401.12345v1"
        # old-style ids keep their archive: ".../abs/hep-th/9901001v1" -> "hep-th/9901001v1"
        arxiv_id = text("id").split("/abs/")[-1]

        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=" ".join(text("title").split()),
            authors=authors,
            abstract=" ".join(text("summary").split()),
            published=datetime.fromisoformat(text("published")),
            updated=datetime.fromisoformat(text("updated")),
            categories=categories,
            pdf_url=pdf_link,  # None fails validation -> the entry is skipped
            primary_category=categories[0] if categories else "Unknown",
        )

    @_retry_transient
    async def download_pdf(self, paper: PaperMetadata, save_path: str) -> bool:
        """Download to `<path>.part`, verify it is a PDF, then atomically rename.

        A crash mid-download must never leave a truncated file at `save_path`, because the runner
        treats an existing file as "already downloaded".
        """
        logger.info("Downloading PDF", arxiv_id=paper.arxiv_id)
        tmp_path = f"{save_path}.part"
        try:
            async with self._throttled():
                async with self._client.stream("GET", paper.pdf_url) as resp:
                    resp.raise_for_status()
                    await self._write_stream(resp, tmp_path)
            await asyncio.to_thread(os.replace, tmp_path, save_path)
        except Exception as e:
            with contextlib.suppress(OSError):
                await asyncio.to_thread(os.remove, tmp_path)
            if isinstance(e, FetchError):
                raise
            raise FetchError(
                f"PDF download failed: {e}",
                code="FETCH_PDF_ERROR",
                metadata={"arxiv_id": paper.arxiv_id, "retryable": _is_transient(e)},
            ) from e

        logger.info("PDF downloaded", arxiv_id=paper.arxiv_id, path=save_path)
        return True

    async def _write_stream(self, resp: httpx.Response, path: str) -> None:
        f = await asyncio.to_thread(open, path, "wb")
        try:
            first = True
            async for chunk in resp.aiter_bytes():
                if first and not chunk.startswith(b"%PDF"):
                    # arXiv sometimes answers 200 with an HTML page (rate limit / withdrawn paper)
                    raise FetchError(
                        "Response is not a PDF",
                        code="FETCH_PDF_ERROR",
                        metadata={"retryable": False},
                    )
                first = False
                await asyncio.to_thread(f.write, chunk)
            if first:
                raise FetchError(
                    "Empty PDF response", code="FETCH_PDF_ERROR", metadata={"retryable": True}
                )
        finally:
            await asyncio.to_thread(f.close)
