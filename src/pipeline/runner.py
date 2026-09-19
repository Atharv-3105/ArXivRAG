import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.core.config import settings
from src.core.exceptions import FetchError, IndexError, ParseError
from src.core.logging import setup_logging
from src.pipeline.chunker import TextChunker
from src.pipeline.fetcher import ArxivFetcher
from src.pipeline.models import PaperMetadata, ProcessedPaper
from src.pipeline.parser import PDFParser
from src.storage.database import dispose_engine, get_async_session, init_db_schema
from src.storage.repositories import PgRepo
from src.storage.vector_db import VectorStore

logger = structlog.get_logger(__name__)

MIN_CHUNKS = 2


@dataclass
class PipelineMetrics:
    fetched: int = 0
    parsed: int = 0
    indexed: int = 0  # chunks written
    failed: int = 0
    skipped: int = 0


async def _record_failure(repo: PgRepo, arxiv_id: str, reason: str) -> None:
    """Best effort: a broken DLQ write must never take the whole run down."""
    try:
        async with get_async_session() as session:
            await repo.log_failure_to_dlq(session, arxiv_id, reason)
    except Exception as e:
        logger.error("Could not write to DLQ", arxiv_id=arxiv_id, error=str(e))


async def run_pipeline(
    query: str | None = None,
    output_dir: str | None = None,
    max_papers: int | None = None,
) -> PipelineMetrics:
    query = query or settings.ARXIV_QUERY
    max_papers = max_papers or settings.ARXIV_MAX_RESULTS
    pdf_dir = Path(output_dir or settings.PDF_DIR)
    await asyncio.to_thread(pdf_dir.mkdir, parents=True, exist_ok=True)

    metrics = PipelineMetrics()

    # Storage
    await init_db_schema()
    vector_store = VectorStore()
    await vector_store.initialize()
    db_repo = PgRepo()

    # Pipeline components
    fetcher = ArxivFetcher()
    parser = PDFParser()
    chunker = TextChunker()

    logger.info("Pipeline started", query=query, max_papers=max_papers)

    try:
        papers = await fetcher.fetch_papers(query=query, max_results=max_papers)
        metrics.fetched = len(papers)

        for paper in papers:
            await _process_paper(
                paper, pdf_dir, fetcher, parser, chunker, db_repo, vector_store, metrics
            )
    finally:
        logger.info("Pipeline finished", metrics=vars(metrics))
        await fetcher.aclose()
        await vector_store.close()
        await dispose_engine()

    return metrics


async def _process_paper(
    paper: PaperMetadata,
    pdf_dir: Path,
    fetcher: ArxivFetcher,
    parser: PDFParser,
    chunker: TextChunker,
    db_repo: PgRepo,
    vector_store: VectorStore,
    metrics: PipelineMetrics,
) -> None:
    """One paper end to end. Failures are isolated: they go to the DLQ, never abort the run."""
    try:
        logger.info("Processing paper", arxiv_id=paper.arxiv_id)
        # old-style ids contain a slash ("hep-th/9901001v1")
        pdf_path = pdf_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"

        if not await asyncio.to_thread(pdf_path.exists):
            await fetcher.download_pdf(paper, str(pdf_path))

        # PyMuPDF is synchronous and CPU-bound; keep it off the event loop.
        sections, _ = await asyncio.to_thread(parser.parse, str(pdf_path))
        metrics.parsed += 1

        chunks = chunker.chunk_sections(paper.arxiv_id, sections)
        if len(chunks) < MIN_CHUNKS:
            logger.warning("Skipping: insufficient chunks", arxiv_id=paper.arxiv_id)
            metrics.skipped += 1
            return

        processed = ProcessedPaper(metadata=paper, sections=sections, chunks=chunks)

        # NOTE: Postgres and Qdrant cannot share a transaction. Postgres is written first, so a
        # Qdrant failure leaves a paper that a re-run will skip (see the xfail test in
        # tests/test_runner.py for the open design question).
        async with get_async_session() as session:
            inserted = await db_repo.upsert_paper_and_chunks(session, processed)

        if inserted:
            await vector_store.upsert_vectors(processed)
            metrics.indexed += len(chunks)
            logger.info("Paper indexed", arxiv_id=paper.arxiv_id, chunks=len(chunks))
        else:
            metrics.skipped += 1

    except (FetchError, ParseError, IndexError) as e:
        metrics.failed += 1
        e.log_error(arxiv_id=paper.arxiv_id)
        await _record_failure(db_repo, paper.arxiv_id, f"{e.code}: {e.message}")

    except Exception as e:
        metrics.failed += 1
        logger.error("Unexpected pipeline error", arxiv_id=paper.arxiv_id, error=str(e))
        await _record_failure(db_repo, paper.arxiv_id, f"UNKNOWN: {e}")


def main() -> None:
    cli = argparse.ArgumentParser(description="Ingest arXiv papers into Postgres + Qdrant")
    cli.add_argument("--query", help="arXiv search_query (default: ARXIV_QUERY from settings)")
    cli.add_argument("--max-papers", type=int, help="default: ARXIV_MAX_RESULTS")
    cli.add_argument("--output-dir", help="where PDFs are cached (default: PDF_DIR)")
    args = cli.parse_args()

    setup_logging()
    asyncio.run(run_pipeline(args.query, args.output_dir, args.max_papers))


if __name__ == "__main__":
    main()
