import os 
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, text 
from datetime import datetime 
from pathlib import Path 

DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_async_engine(DATABASE_URL, echo = False)
AsyncSessionLocal = async_sessionmaker(engine, class_ = AsyncSession, expire_on_commit = False)

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

async def init_db_schema() -> None:
    schema_path = Path(__file__).resolve().parent.parent.parent / "storage" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema Missing at :{schema_path}")
    
    sql_script = schema_path.read_text(encoding = "utf-8")
    
    async with engine.begin() as conn:
        await conn.execute(text(sql_script))

    print("✅ Database schema initialized & verified (idempotent).")