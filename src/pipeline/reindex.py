"""Rebuild the vector index from Postgres, the source of truth.

    python -m src.pipeline.reindex          # repair papers stuck in 'pending' (the reconciler)
    python -m src.pipeline.reindex --all    # re-embed every paper (new model / new collection)

Because Qdrant is a derived index, it can be dropped and rebuilt at any time without touching arXiv:
this is how a collection migration (e.g. dense-only v1 -> hybrid v2) is done.
"""

import argparse
import asyncio

import structlog

from src.core.logging import setup_logging
from src.pipeline.runner import index_from_store
from src.storage.database import dispose_engine, get_async_session, init_db_schema
from src.storage.repositories import PgRepo
from src.storage.vector_db import VectorStore

logger = structlog.get_logger(__name__)


async def reindex(all_papers: bool = False) -> dict:
    await init_db_schema()
    repo = PgRepo()
    store = VectorStore()
    await store.initialize()
    report = {"papers": 0, "chunks": 0, "failed": 0}
    try:
        async with get_async_session() as session:
            ids = (
                await repo.all_paper_ids(session)
                if all_papers
                else await repo.pending_papers(session, limit=10_000)
            )
        logger.info("Reindex started", papers=len(ids), mode="all" if all_papers else "pending")
        for arxiv_id in ids:
            try:
                report["chunks"] += await index_from_store(repo, store, arxiv_id)
                report["papers"] += 1
            except Exception as e:
                report["failed"] += 1
                logger.error("Reindex failed", arxiv_id=arxiv_id, error=str(e))
        if report["papers"]:
            from src.pipeline.runner import _refresh_influence

            await _refresh_influence()
    finally:
        await store.close()
        await dispose_engine()
    logger.info("Reindex finished", **report)
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description="Rebuild Qdrant from Postgres")
    cli.add_argument("--all", action="store_true", help="re-embed every paper, not just pending")
    args = cli.parse_args()
    setup_logging()
    asyncio.run(reindex(args.all))


if __name__ == "__main__":
    main()
