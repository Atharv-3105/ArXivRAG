from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = Field(default="ArXivMind", description="Service name")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging Level")
    SECRET_KEY: str = Field(..., description="Required for sessions/auth")

    # API
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    CORS_ORIGINS: list[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])

    # Storage
    DATABASE_URL: str | None = Field(default=None, description="PostgreSQL connection string")
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant API endpoint")
    QDRANT_API_KEY: str | None = Field(default=None, description="Only for secured/cloud Qdrant")
    QDRANT_COLLECTION: str = Field(default="arxiv_papers_v1")

    # Embeddings
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")
    EMBEDDING_DIM: int = Field(default=384, description="Must match EMBEDDING_MODEL's output size")
    EMBEDDING_BATCH_SIZE: int = Field(default=32)

    # ArXiv ingestion
    ARXIV_QUERY: str = Field(
        default='(cat:cs.CL OR cat:cs.LG OR cat:cs.AI) AND abs:"large language models"',
        description="arXiv search_query: use field prefixes + quoted phrases, not bare words",
    )
    ARXIV_MAX_RESULTS: int = Field(default=5, ge=1, le=100)
    ARXIV_SORT_BY: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "submittedDate"
    ARXIV_REQUEST_DELAY_SECONDS: float = Field(
        default=3.0, description="Pause between arXiv requests (arXiv asks for ~1 per 3s)"
    )
    PDF_DIR: str = Field(default="data/papers")

    # Parsing / chunking
    PARSE_MIN_WORDS: int = Field(default=800)
    PARSE_MIN_SECTIONS: int = Field(default=2)
    CHUNK_SIZE: int = Field(default=1000, description="Characters per chunk")
    CHUNK_OVERLAP: int = Field(default=200, description="Characters shared between chunks")

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return v.upper()

    @field_validator("DATABASE_URL")
    @classmethod
    def use_async_driver(cls, v: str | None) -> str | None:
        """SQLAlchemy's async engine needs asyncpg; accept the common `postgresql://` spelling."""
        if v and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


settings = Settings()
