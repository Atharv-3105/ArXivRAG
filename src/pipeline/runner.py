import argparse
import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

from src.core.config import settings
from src.core.exceptions import FetchError, IndexError, ParseError
from src.core.logging import setup_logging
from src.pipeline.chunker import TextChunker
from src.pipeline.extractor import PaperExtractor
from src.pipeline.fetcher import ArxivFetcher
from src.pipeline.models import PaperMetadata, ProcessedPaper
from src.storage.database import dispose_engine, get_async_session, init_db_schema
from src.storage.repositories import INDEXED, PgRepo
from src.storage.vector_db import VectorStore

logger = structlog.get_logger(__name__)

MIN_CHUNKS = 2


@dataclass
class PipelineMetrics:
    fetched: int = 0
    parsed: int = 0
    indexed: int = 0  # chunks written
    failed: int = 0
    skipped: int = 0  # already indexed: no download, no parse
    repaired: int = 0  # stored but never indexed (crash between the two stores): re-embedded
    html: int = 0  # papers parsed from arXiv HTML (formulas are real LaTeX)
    pdf: int = 0  # papers that fell back to PDF (formulas are degraded)

    @property
    def parse_success_rate(self) -> float | None:
        attempts = self.parsed + self.failed
        return round(self.parsed / attempts, 3) if attempts else None


async def _record_failure(repo: PgRepo, arxiv_id: str, reason: str) -> None:
    """Best effort: a broken DLQ write must never take the whole run down."""
    try:
        async with get_async_session() as session:
            await repo.log_failure_to_dlq(session, arxiv_id, reason)
    except Exception as e:
        logger.error("Could not write to DLQ", arxiv_id=arxiv_id, error=str(e))


async def _refresh_influence() -> None:
    """New papers change the citation graph: recompute PageRank. Best effort."""
    from src.graph.influence import refresh_influence
    from src.graph.store import PgGraphStore

    try:
        await refresh_influence(PgGraphStore())
    except Exception as e:
        logger.error("Influence refresh failed", error=str(e))


async def index_from_store(repo: PgRepo, vector_store: VectorStore, arxiv_id: str) -> int:
    """Re-embed a paper from Postgres (the source of truth) and mark it indexed. Returns chunks."""
    async with get_async_session() as session:
        processed = await repo.load_processed(session, arxiv_id)
    if processed is None:
        return 0
    await vector_store.upsert_vectors(processed)
    async with get_async_session() as session:
        await repo.mark_indexed(session, arxiv_id)
    return len(processed.chunks)


async def run_pipeline(
    query: str | None = None,
    cache_dir: str | None = None,
    max_papers: int | None = None,
    reingest: bool = False,
    ids: list[str] | None = None,
) -> PipelineMetrics:
    query = None if ids and not query else (query or settings.ARXIV_QUERY)
    max_papers = max_papers or settings.ARXIV_MAX_RESULTS
    cache = Path(cache_dir or settings.CACHE_DIR)

    metrics = PipelineMetrics()

    # Storage
    await init_db_schema()
    vector_store = VectorStore()
    await vector_store.initialize()
    db_repo = PgRepo()
    async with get_async_session() as session:
        run_id = await db_repo.start_run(session, query)

    # Pipeline components
    fetcher = ArxivFetcher()
    extractor = PaperExtractor(fetcher, cache)
    chunker = TextChunker(token_counter=vector_store.embedder.token_counter())

    logger.info("Pipeline started", run_id=run_id, query=query, max_papers=max_papers)
    status = "failed"
    try:
        papers = await fetcher.fetch_papers(query=query, max_results=max_papers, id_list=ids)
        metrics.fetched = len(papers)

        for paper in papers:
            await _process_paper(
                paper, extractor, chunker, db_repo, vector_store, metrics, reingest
            )
        status = "succeeded" if metrics.failed == 0 else "partial"
        if metrics.indexed:
            await _refresh_influence()
    finally:
        report = {**asdict(metrics), "parse_success_rate": metrics.parse_success_rate}
        logger.info("Pipeline finished", run_id=run_id, status=status, metrics=report)
        try:
            async with get_async_session() as session:
                await db_repo.finish_run(session, run_id, status, report)
        except Exception as e:
            logger.error("Could not record ingestion run", run_id=run_id, error=str(e))
        await fetcher.aclose()
        await vector_store.close()
        await dispose_engine()

    return metrics


async def _process_paper(
    paper: PaperMetadata,
    extractor: PaperExtractor,
    chunker: TextChunker,
    db_repo: PgRepo,
    vector_store: VectorStore,
    metrics: PipelineMetrics,
    reingest: bool = False,
) -> None:
    """One paper end to end. Failures are isolated: they go to the DLQ, never abort the run.

    Consistency contract (Postgres = source of truth, Qdrant = derived index):
      not stored          -> parse, write Postgres as 'pending', write Qdrant, mark 'indexed'
      stored, 'pending'   -> a previous run died between the two writes: re-embed from Postgres
      stored, 'indexed'   -> nothing to do (no download, no parse)
    """
    try:
        logger.info("Processing paper", arxiv_id=paper.arxiv_id)
        async with get_async_session() as session:
            status = await db_repo.get_index_status(session, paper.arxiv_id)

        if status is not None and not reingest:
            if status == INDEXED:
                metrics.skipped += 1
                logger.info("Already indexed, skipping", arxiv_id=paper.arxiv_id)
                return
            repaired_chunks = await index_from_store(db_repo, vector_store, paper.arxiv_id)
            metrics.repaired += 1
            metrics.indexed += repaired_chunks
            logger.info("Repaired pending paper from Postgres", arxiv_id=paper.arxiv_id)
            return

        doc = await extractor.extract(paper)
        metrics.parsed += 1
        if doc.method == "html":
            metrics.html += 1
        else:
            metrics.pdf += 1

        chunks = chunker.chunk_sections(paper.arxiv_id, doc.sections)
        if len(chunks) < MIN_CHUNKS:
            logger.warning("Skipping: insufficient chunks", arxiv_id=paper.arxiv_id)
            metrics.skipped += 1
            return

        processed = ProcessedPaper(
            metadata=paper,
            sections=doc.sections,
            chunks=chunks,
            equations=doc.equations,
            references=doc.references,
            citation_contexts=doc.citation_contexts,
            parse_method=doc.method,
            math_fidelity=doc.math_fidelity,
            parse_stats=doc.stats,
        )

        async with get_async_session() as session:
            superseded = await db_repo.other_versions(session, paper.arxiv_id)
        for old_id in superseded:
            # arXiv served a different version than the one stored. Keeping both would index the
            # same passages twice, and they would be cited as two different sources.
            async with get_async_session() as session:
                await db_repo.delete_paper(session, old_id)
            await vector_store.delete_paper(old_id)
            logger.info("Older version superseded", old=old_id, new=paper.arxiv_id)

        if reingest and status is not None:
            # Only after the new parse succeeded, so a bad re-parse never destroys good data.
            async with get_async_session() as session:
                await db_repo.delete_paper(session, paper.arxiv_id)
            await vector_store.delete_paper(paper.arxiv_id)

        async with get_async_session() as session:
            inserted = await db_repo.upsert_paper_and_chunks(session, processed)
        if not inserted:  # a concurrent run stored it between our check and our insert
            metrics.skipped += 1
            return

        await vector_store.upsert_vectors(processed)
        async with get_async_session() as session:
            await db_repo.mark_indexed(session, paper.arxiv_id)
        metrics.indexed += len(chunks)
        logger.info(
            "Paper indexed",
            arxiv_id=paper.arxiv_id,
            chunks=len(chunks),
            method=doc.method,
            equations=len(doc.equations),
        )

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
    cli.add_argument("--ids", help="comma-separated arXiv ids to fetch exactly (overrides query)")
    cli.add_argument("--cache-dir", help="where HTML/PDF are cached (default: CACHE_DIR)")
    cli.add_argument(
        "--reingest",
        action="store_true",
        help="replace papers that are already stored (e.g. to upgrade a PDF parse to HTML)",
    )
    args = cli.parse_args()

    setup_logging()
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    asyncio.run(run_pipeline(args.query, args.cache_dir, args.max_papers, args.reingest, ids))


if __name__ == "__main__":
    main()
