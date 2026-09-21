"""Chooses how to turn an arXiv paper into structured text.

Tier 1  arXiv HTML  : real LaTeX for every formula, reliable section tree, structured references.
Tier 2  PDF         : universal fallback (column-aware reading order); math is degraded.

Every result records which tier produced it and why an earlier tier was skipped, so parse
quality can be measured per paper instead of guessed.
"""

import asyncio
import contextlib
import os
from pathlib import Path

import structlog

from src.core.exceptions import FetchError, ParseError
from src.pipeline.fetcher import ArxivFetcher
from src.pipeline.html_parser import HtmlParser
from src.pipeline.models import PaperMetadata, ParsedDocument
from src.pipeline.parser import PDFParser

logger = structlog.get_logger(__name__)


def _safe_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")  # old-style ids contain a slash ("hep-th/9901001v1")


class PaperExtractor:
    def __init__(
        self,
        fetcher: ArxivFetcher,
        cache_dir: Path,
        html_parser: HtmlParser | None = None,
        pdf_parser: PDFParser | None = None,
    ):
        self.fetcher = fetcher
        self.html_dir = cache_dir / "html"
        self.pdf_dir = cache_dir / "pdf"
        self.html_parser = html_parser or HtmlParser()
        self.pdf_parser = pdf_parser or PDFParser()

    async def extract(self, paper: PaperMetadata) -> ParsedDocument:
        await asyncio.to_thread(self.html_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(self.pdf_dir.mkdir, parents=True, exist_ok=True)

        fallback_reason = await self._try_html(paper)
        if isinstance(fallback_reason, ParsedDocument):
            return fallback_reason

        logger.info("Falling back to PDF", arxiv_id=paper.arxiv_id, reason=fallback_reason)
        pdf_path = self.pdf_dir / f"{_safe_name(paper.arxiv_id)}.pdf"
        if not await asyncio.to_thread(pdf_path.exists):
            await self.fetcher.download_pdf(paper, str(pdf_path))

        # PyMuPDF is synchronous and CPU-bound; keep it off the event loop.
        sections, metrics = await asyncio.to_thread(self.pdf_parser.parse, str(pdf_path))
        return ParsedDocument(
            method="pdf",
            math_fidelity="degraded",
            sections=sections,
            stats={**vars(metrics), "html_fallback_reason": fallback_reason},
        )

    async def _try_html(self, paper: PaperMetadata) -> ParsedDocument | str:
        """A ParsedDocument on success, otherwise the reason the HTML tier was skipped."""
        path = self.html_dir / f"{_safe_name(paper.arxiv_id)}.html"
        try:
            if await asyncio.to_thread(path.exists):
                html_text = await asyncio.to_thread(path.read_text, "utf-8")
            else:
                fetched = await self.fetcher.fetch_html(paper)
                if fetched is None:
                    return "no_html"
                html_text = fetched
                await asyncio.to_thread(_atomic_write, path, html_text)
            return await asyncio.to_thread(self.html_parser.parse, html_text)
        except ParseError as e:
            logger.warning("HTML parse rejected", arxiv_id=paper.arxiv_id, code=e.code)
            return e.code
        except FetchError as e:
            # HTML is an optimisation, not a requirement: if arXiv's HTML host is unhappy, the PDF
            # tier can still ingest the paper.
            logger.warning("HTML fetch failed", arxiv_id=paper.arxiv_id, code=e.code)
            return e.code


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
