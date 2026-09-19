import structlog
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert  # the generic insert has no ON CONFLICT
from sqlalchemy.ext.asyncio import AsyncSession

from src.pipeline.models import ProcessedPaper
from src.storage.database import FailedPaper, Paper, PaperChunk

logger = structlog.get_logger(__name__)


class PgRepo:
    async def upsert_paper_and_chunks(
        self, session: AsyncSession, paper_data: ProcessedPaper
    ) -> bool:
        """Insert paper + chunks in ONE transaction. Returns False if the paper already exists."""
        paper_meta = paper_data.metadata

        result = await session.execute(select(Paper).where(Paper.arxiv_id == paper_meta.arxiv_id))
        if result.scalars().first():
            logger.info("Paper already exists, skipping", arxiv_id=paper_meta.arxiv_id)
            return False

        paper_stmt = (
            insert(Paper)
            .values(
                arxiv_id=paper_meta.arxiv_id,
                title=paper_meta.title,
                abstract=paper_meta.abstract,
                authors=paper_meta.authors,
                published_date=paper_meta.published,
                pdf_url=paper_meta.pdf_url,
                categories=paper_meta.categories,
            )
            .on_conflict_do_nothing(index_elements=["arxiv_id"])
        )
        await session.execute(paper_stmt)

        chunk_values = [
            {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "section_title": c.section_title,
                "content": c.content,
                "chunk_metadata": c.metadata,
            }
            for c in paper_data.chunks
        ]
        chunk_stmt = (
            insert(PaperChunk)
            .values(chunk_values)
            .on_conflict_do_nothing(index_elements=["chunk_id"])
        )
        await session.execute(chunk_stmt)

        # Nothing is visible until this commit; any exception above leaves no half-written paper.
        await session.commit()

        logger.info(
            "Paper & chunks upserted", arxiv_id=paper_meta.arxiv_id, chunk_count=len(chunk_values)
        )
        return True

    async def log_failure_to_dlq(self, session: AsyncSession, arxiv_id: str, reason: str) -> None:
        """Record a failed ingestion (upsert: repeated failures bump retry_count)."""
        stmt = (
            insert(FailedPaper)
            .values(arxiv_id=arxiv_id, failure_reason=reason, retry_count=0)
            .on_conflict_do_update(
                index_elements=["arxiv_id"],
                set_={
                    "failure_reason": reason,
                    "retry_count": FailedPaper.retry_count + 1,
                    "attempted_at": text("NOW()"),
                },
            )
        )
        await session.execute(stmt)
        await session.commit()

        logger.warning("Failed paper added to DLQ", arxiv_id=arxiv_id, reason=reason)
