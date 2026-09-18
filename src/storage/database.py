import os 
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy import Column, String,Integer, Text, DateTime, JSON, ForeignKey, text 
from sqlalchemy.pool import AsyncAdaptedQueuePool
import structlog 
from pathlib import Path 

logger = structlog.get_logger(__name__)


DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_async_engine(DATABASE_URL, 
                             echo = False,
                             poolclass = AsyncAdaptedQueuePool,
                             pool_size = 10,
                             max_overflow = 20,
                             pool_timeout = 30,
                             pool_recycle = 1800,
        )

AsyncSessionLocal = async_sessionmaker(engine, class_ = AsyncSession, expire_on_commit = False, autoflush = False)

class Base(DeclarativeBase):
    pass 

#ORM Models: Maps to schema.sql tables
class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = {"schema": "public"}
    
    arxiv_id = Column(String(20), primary_key = True, index = True)
    title  = Column(Text, nullable = False)
    abstract = Column(Text, nullable=False)
    authors = Column(JSON, nullable=False)
    published_date = Column(DateTime(timezone=True), nullable=False)
    pdf_url = Column(Text)
    categories = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    
    chunks = relationship("PaperChunk", back_populates="paper", cascade="all, delete-orphan")

class PaperChunk(Base):
    __tablename__ = "paper_chunks"
    __table_args__ = {"schema": "public"}
    
    chunk_id = Column(String(100), primary_key=True)
    paper_id = Column(String(20), ForeignKey("papers.arxiv_id", ondelete="CASCADE"), nullable=False)
    section_title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    
    paper = relationship("Paper", back_populates="chunks")
    
class FailedPaper(Base):
    __tablename__ = "failed_papers"
    arxiv_id = Column(String(20), primary_key=True)
    failure_reason = Column(Text, nullable = False)
    attempted_at = Column(DateTime(timezone = True), server_default = text("NOW()"))
    retry_count = Column(Integer, server_default = text("0"))

async def init_db_schema() -> None:
    schema_path = Path(__file__).resolve().parent.parent.parent / "storage" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema Missing at :{schema_path}")
    
    sql_script = schema_path.read_text(encoding = "utf-8")
    
    async with engine.begin() as conn:
        await conn.execute(text(sql_script))

    logger.info("Database schema initialized & verified")

async def get_async_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session