from datetime import UTC, datetime

import structlog
from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert  # the generic insert has no ON CONFLICT
from sqlalchemy.ext.asyncio import AsyncSession

from src.pipeline.citations import citation_edges, context_rows
from src.pipeline.models import Chunk, PaperMetadata, ProcessedPaper
from src.storage.database import (
    CitationContextRow,
    FailedPaper,
    IngestionRun,
    Paper,
    PaperChunk,
    PaperCitation,
    PaperEquation,
    PaperReference,
)

INDEXED = "indexed"
PENDING = "pending"

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
                parse_method=paper_data.parse_method,
                math_fidelity=paper_data.math_fidelity,
                parse_stats=paper_data.parse_stats,
                # 'indexed' only after Qdrant confirms the vectors (see mark_indexed)
                index_status=PENDING,
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

        if paper_data.equations:
            eq_rows = [
                {"paper_id": paper_meta.arxiv_id, **eq.model_dump()} for eq in paper_data.equations
            ]
            await session.execute(insert(PaperEquation).values(eq_rows).on_conflict_do_nothing())
        if paper_data.references:
            ref_rows = [
                {
                    "paper_id": paper_meta.arxiv_id,
                    "ordinal": r.ordinal,
                    "ref_key": r.key,
                    "text": r.text,
                    "urls": r.urls,
                }
                for r in paper_data.references
            ]
            await session.execute(insert(PaperReference).values(ref_rows).on_conflict_do_nothing())

        edges = citation_edges(paper_meta.arxiv_id, paper_data.references)
        if edges:
            await session.execute(insert(PaperCitation).values(edges).on_conflict_do_nothing())

        contexts = context_rows(
            paper_meta.arxiv_id, paper_data.references, paper_data.citation_contexts
        )
        if contexts:
            await session.execute(insert(CitationContextRow).values(contexts))

        # Nothing is visible until this commit; any exception above leaves no half-written paper.
        await session.commit()

        logger.info(
            "Paper & chunks upserted", arxiv_id=paper_meta.arxiv_id, chunk_count=len(chunk_values)
        )
        return True

    async def get_index_status(self, session: AsyncSession, arxiv_id: str) -> str | None:
        """'indexed' | 'pending' | None (not stored)."""
        result = await session.execute(select(Paper.index_status).where(Paper.arxiv_id == arxiv_id))
        return result.scalar_one_or_none()

    async def mark_indexed(self, session: AsyncSession, arxiv_id: str) -> None:
        """Flip to 'indexed' and clear any dead-letter entry: the paper has now succeeded."""
        await session.execute(
            update(Paper)
            .where(Paper.arxiv_id == arxiv_id)
            .values(index_status=INDEXED, updated_at=datetime.now(UTC))
        )
        await session.execute(delete(FailedPaper).where(FailedPaper.arxiv_id == arxiv_id))
        await session.commit()

    async def pending_papers(self, session: AsyncSession, limit: int = 100) -> list[str]:
        result = await session.execute(
            select(Paper.arxiv_id)
            .where(Paper.index_status != INDEXED)
            .order_by(Paper.created_at)
            .limit(limit)
        )
        return list(result.scalars())

    async def all_paper_ids(self, session: AsyncSession) -> list[str]:
        result = await session.execute(select(Paper.arxiv_id).order_by(Paper.created_at))
        return list(result.scalars())

    async def load_processed(self, session: AsyncSession, arxiv_id: str) -> ProcessedPaper | None:
        """Rebuild what the vector store needs from Postgres, the source of truth.

        This is what makes Qdrant a *derived* index: any paper can be re-embedded from here without
        re-downloading or re-parsing it.
        """
        paper = (
            await session.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
        ).scalar_one_or_none()
        if paper is None:
            return None
        rows = (
            await session.execute(
                select(PaperChunk)
                .where(PaperChunk.paper_id == arxiv_id)
                .order_by(PaperChunk.created_at)
            )
        ).scalars()
        chunks = [
            Chunk(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                section_title=r.section_title,
                content=r.content,
                context=(r.chunk_metadata or {}).get("section_path") or r.section_title,
                metadata=r.chunk_metadata or {},
            )
            for r in rows
        ]
        metadata = PaperMetadata(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            published=paper.published_date,
            updated=paper.updated_at or paper.published_date,
            categories=paper.categories or [],
            pdf_url=paper.pdf_url or "",
            primary_category=(paper.categories or ["Unknown"])[0],
        )
        return ProcessedPaper(
            metadata=metadata,
            sections=[],
            chunks=chunks,
            parse_method=paper.parse_method or "pdf",
            math_fidelity=paper.math_fidelity or "degraded",
        )

    async def start_run(self, session: AsyncSession, query: str | None) -> int:
        result = await session.execute(
            insert(IngestionRun).values(query=query).returning(IngestionRun.id)
        )
        run_id = result.scalar_one()
        await session.commit()
        return run_id

    async def finish_run(
        self, session: AsyncSession, run_id: int, status: str, metrics: dict
    ) -> None:
        await session.execute(
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(status=status, metrics=metrics, finished_at=datetime.now(UTC))
        )
        await session.commit()

    async def delete_paper(self, session: AsyncSession, arxiv_id: str) -> None:
        """Delete a paper; chunks, equations and references go with it (ON DELETE CASCADE)."""
        await session.execute(delete(Paper).where(Paper.arxiv_id == arxiv_id))
        await session.commit()
        logger.info("Paper deleted for re-ingestion", arxiv_id=arxiv_id)

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
