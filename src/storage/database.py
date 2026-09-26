from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import structlog
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import AsyncAdaptedQueuePool

from src.core.config import settings

logger = structlog.get_logger(__name__)

if not settings.DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Expected e.g. "
        "postgresql+asyncpg://USER:PASS@localhost:5433/arxivmind (see .env.example)"
    )

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    poolclass=AsyncAdaptedQueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


# ORM models mirror schema.sql (which is applied by init_db_schema).
class Paper(Base):
    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(String(30), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list] = mapped_column(JSONB, nullable=False)
    published_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    categories: Mapped[list | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    parse_method: Mapped[str | None] = mapped_column(String(16))
    math_fidelity: Mapped[str | None] = mapped_column(String(16))
    parse_stats: Mapped[dict | None] = mapped_column(JSONB)
    index_status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'indexed'")
    )
    base_id: Mapped[str | None] = mapped_column(
        String(30), Computed("regexp_replace(arxiv_id, 'v[0-9]+$', '')", persisted=True)
    )

    chunks = relationship("PaperChunk", back_populates="paper", cascade="all, delete-orphan")


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    chunk_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        String(30), ForeignKey("papers.arxiv_id", ondelete="CASCADE"), nullable=False
    )
    section_title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # `metadata` is reserved on declarative classes, so the attribute has another name
    # while the database column keeps its name.
    chunk_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )

    paper = relationship("Paper", back_populates="chunks")


class PaperEquation(Base):
    __tablename__ = "paper_equations"

    paper_id: Mapped[str] = mapped_column(
        String(30), ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    latex: Mapped[str] = mapped_column(Text, nullable=False)
    number: Mapped[str | None] = mapped_column(String(32))
    display: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("TRUE"))
    section_path: Mapped[str | None] = mapped_column(Text)
    anchor: Mapped[str | None] = mapped_column(String(100))


class PaperReference(Base):
    __tablename__ = "paper_references"

    paper_id: Mapped[str] = mapped_column(
        String(30), ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    ref_key: Mapped[str | None] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    urls: Mapped[list | None] = mapped_column(JSONB)


class PaperCitation(Base):
    __tablename__ = "paper_citations"

    citing_id: Mapped[str] = mapped_column(
        String(30), ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    cited_base_id: Mapped[str] = mapped_column(String(30), primary_key=True)
    ref_ordinal: Mapped[int | None] = mapped_column(Integer)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    query: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), server_default=text("'running'"))
    metrics: Mapped[dict | None] = mapped_column(JSONB)


class FailedPaper(Base):
    __tablename__ = "failed_papers"

    arxiv_id: Mapped[str] = mapped_column(String(30), primary_key=True)
    failure_reason: Mapped[str] = mapped_column(Text, nullable=False)
    attempted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    retry_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))


SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _split_statements(sql_script: str) -> list[str]:
    """asyncpg cannot run several commands in one prepared statement, so run them one by one."""
    lines = [ln for ln in sql_script.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


async def init_db_schema() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema missing at: {SCHEMA_PATH}")

    statements = _split_statements(SCHEMA_PATH.read_text(encoding="utf-8"))
    async with engine.begin() as conn:  # one transaction: schema is all-or-nothing
        for statement in statements:
            await conn.execute(text(statement))

    logger.info("Database schema initialized & verified", statements=len(statements))


@asynccontextmanager
async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Usage: `async with get_async_session() as session:`"""
    async with AsyncSessionLocal() as session:
        yield session


async def dispose_engine() -> None:
    await engine.dispose()
