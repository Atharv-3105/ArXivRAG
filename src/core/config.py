from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every tunable lives here (plan §8: config-driven behaviour, no hardcoded values)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = Field(default="ArXivMind", description="Service name")
    APP_VERSION: str = Field(default="0.2.0")
    ENVIRONMENT: Literal["dev", "test", "prod"] = "dev"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging Level")
    SECRET_KEY: str = Field(..., description="Signs JWTs. Required; use >=32 random chars")
    SECRET_KEY_PREVIOUS: str | None = Field(
        default=None, description="Rotation window: tokens signed with it still verify"
    )

    # API
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    CORS_ORIGINS: list[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    PUBLIC_BASE_URL: str = Field(default="http://localhost:8000")
    FRONTEND_DIST: str = Field(
        default="frontend/dist", description="Built web app to serve at / (relative to the repo)"
    )

    # Storage
    DATABASE_URL: str | None = Field(default=None, description="PostgreSQL connection string")
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant API endpoint")
    QDRANT_API_KEY: str | None = Field(default=None, description="Only for secured/cloud Qdrant")
    QDRANT_COLLECTION: str = Field(
        default="arxiv_papers_minilm",
        description="One collection per embedding model: a model swap never mixes vector spaces",
    )
    DEPENDENCY_TIMEOUT_SECONDS: float = Field(default=3.0, description="Readiness probe timeout")

    # Embeddings (fastembed / ONNX: no PyTorch, fits free-tier memory)
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = Field(default=384, description="Must match EMBEDDING_MODEL's output size")
    EMBEDDING_BATCH_SIZE: int = Field(default=32)
    EMBEDDING_QUERY_PREFIX: str = Field(
        default="",
        description="Prepended to queries only. MiniLM is symmetric, so none; bge needs its "
        "'Represent this sentence for searching relevant passages: ' instruction",
    )
    SPARSE_MODEL: str = Field(default="Qdrant/bm25", description="Sparse model for hybrid search")
    MODEL_CACHE_DIR: str = Field(
        default="data/models", description="ONNX model cache (fastembed's default is a temp dir)"
    )

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
    CACHE_DIR: str = Field(
        default="data", description="Downloaded HTML/PDF live in <dir>/html, pdf"
    )

    # Parsing / chunking
    PARSE_MIN_WORDS: int = Field(default=800)
    PARSE_MIN_SECTIONS: int = Field(default=2)
    CHUNK_SIZE: int = Field(
        default=800,
        description="Characters per chunk. MiniLM reads 256 word-pieces and silently drops the "
        "rest; 800 characters of LaTeX-heavy prose stays inside that",
    )
    CHUNK_OVERLAP: int = Field(default=150, description="Characters shared between chunks")

    # Retrieval
    RETRIEVAL_TOP_K: int = Field(default=6, ge=1, le=30)
    RETRIEVAL_PREFETCH: int = Field(default=30, description="Candidates per branch before RRF")
    RETRIEVAL_MAX_PER_PAPER: int = Field(
        default=3, ge=1, description="Chunks per paper in one answer's context (diversity)"
    )
    CHAT_HISTORY_MESSAGES: int = Field(default=6, description="Past messages sent to the LLM")
    FOLLOWUP_MAX_WORDS: int = Field(
        default=12, description="Shorter messages are treated as follow-ups when retrieving"
    )
    GRAPH_CONTEXT_PAPERS: int = Field(default=3, description="Cited papers added as context")
    GRAPH_MAX_DEPTH: int = Field(default=2, ge=1, le=4)
    GRAPH_ENABLED: bool = Field(default=True, description="Graph expansion + influence re-rank")
    GRAPH_TIMEOUT_SECONDS: float = Field(
        default=1.5, description="Budget for all graph work in one turn; then vector-only"
    )
    GRAPH_MAX_NODES: int = Field(default=150, le=500, description="Cap for /papers/{id}/graph")
    MASTERY_WEAK_BELOW: float = Field(default=0.35, ge=0, le=1)
    MASTERY_STRONG_ABOVE: float = Field(default=0.7, ge=0, le=1)
    RERANK_SECTION_BOOST: float = Field(default=1.25, ge=1, description="Level-matched sections")
    RERANK_INFLUENCE_WEIGHT: float = Field(default=0.1, ge=0, description="Per unit of PageRank")
    RATE_LIMIT_PUBLIC_PER_MINUTE: int = Field(default=60, description="Per IP, public endpoints")

    # LLM router: several free-tier providers with per-task priority, failover, rate-limit budgets
    # and circuit breakers (src/llm/router.py). A provider is enabled by setting its key.
    GROQ_API_KEY: SecretStr | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GEMINI_API_KEY: SecretStr | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    CEREBRAS_API_KEY: SecretStr | None = None
    CEREBRAS_MODEL: str = "llama3.3-70b"
    OPENROUTER_API_KEY: SecretStr | None = None
    OPENROUTER_MODELS: list[str] = Field(
        default=["z-ai/glm-5.2:free", "minimax/minimax-m3:free"],
        description="Free models tried in order; they share ONE account-wide rate limit",
    )
    OLLAMA_BASE_URL: str | None = Field(default=None, description="e.g. http://localhost:11434/v1")
    OLLAMA_MODEL: str = "llama3.1:8b"
    ANTHROPIC_API_KEY: SecretStr | None = Field(default=None, description="Paid; off unless set")
    ANTHROPIC_MODEL: str = "claude-opus-5"
    ANTHROPIC_SERVER_FALLBACKS: bool = Field(
        default=True, description="Retry safety declines on a fallback model (server-side)"
    )
    LLM_PROVIDERS_JSON: str | None = Field(
        default=None,
        description="Full override: JSON list of provider configs (see src/llm/registry.py)",
    )
    LLM_ENABLE_MOCK: bool | None = Field(
        default=None, description="Offline mock provider. Default: on only if no real provider"
    )
    LLM_TASK_ORDERS: dict[str, list[str]] = Field(
        default={
            # classification: fastest first
            "router": ["groq", "cerebras", "gemini", "openrouter", "ollama", "anthropic"],
            # teaching: strongest free reasoning first
            "tutor": ["gemini", "openrouter", "groq", "cerebras", "ollama", "anthropic"],
            "default": ["gemini", "groq", "openrouter", "cerebras", "ollama", "anthropic"],
        },
        description="Provider preference per task; names may be prefixes ('openrouter')",
    )
    LLM_MAX_TOKENS: int = Field(default=1500)
    LLM_TEMPERATURE: float = Field(default=0.2, description="Ignored by providers that reject it")
    LLM_TIMEOUT_SECONDS: float = Field(default=60.0, description="Whole non-streaming call")
    LLM_FIRST_TOKEN_TIMEOUT_SECONDS: float = Field(default=20.0)
    LLM_STREAM_IDLE_TIMEOUT_SECONDS: float = Field(default=30.0)
    LLM_BREAKER_FAILURE_THRESHOLD: int = Field(default=3, ge=1)
    LLM_BREAKER_RECOVERY_SECONDS: float = Field(default=120.0)
    LLM_BREAKER_MAX_RECOVERY_SECONDS: float = Field(default=1800.0)
    LLM_DEFAULT_RETRY_AFTER_SECONDS: float = Field(default=60.0)
    ROUTER_MIN_CONFIDENCE: float = Field(
        default=0.6, ge=0, le=1, description="Below this the intent falls back to 'chat'"
    )
    ROUTER_USE_LLM: bool = Field(
        default=False, description="Rule-based intent router is free; LLM router costs one call"
    )

    # Auth / SaaS
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_TTL_MINUTES: int = Field(default=60 * 24)
    BCRYPT_ROUNDS: int = Field(default=12, ge=4, le=16, description="Password hash cost (2^n)")
    ALLOW_REGISTRATION: bool = True
    ADMIN_EMAILS: list[str] = Field(default_factory=list)
    SEED_ADMIN_EMAIL: str | None = Field(
        default=None, description="Owner account created (or promoted) as admin at startup"
    )
    SEED_ADMIN_PASSWORD: SecretStr | None = Field(
        default=None, description="Its password on first creation; never changed afterwards"
    )

    # Free-tier limits (protect the LLM budget and the free hosting quotas)
    RATE_LIMIT_PER_MINUTE: int = Field(default=20, description="Requests per user per minute")
    FREE_DAILY_MESSAGES: int = Field(default=50, description="Chat turns per user per UTC day")
    MAX_MESSAGE_CHARS: int = Field(default=4000)

    # Memory / personalisation
    MEMORY_EMA_ALPHA: float = Field(default=0.3, gt=0, le=1)
    MEMORY_DECAY_HALF_LIFE_DAYS: float = Field(default=30.0, gt=0)
    MEMORY_MAX_UPDATES_PER_MINUTE: int = Field(default=30)
    MEMORY_QUEUE_SIZE: int = Field(default=1000)
    MEMORY_MAX_RETRIES: int = Field(default=3)
    PROFILE_CONCEPTS_IN_PROMPT: int = Field(default=6, description="Mastery entries shown to LLM")

    # Pedagogy / agents (Phase 5)
    PROMPT_AB_TEST: bool = Field(default=True, description="Split prompt variants by user")
    PROMPT_VARIANTS: dict[str, list[str]] = Field(
        default={"tutor_explain": ["v1", "v2"]},
        description="Template versions in rotation per prompt; the first is the default",
    )
    TUTOR_MAX_HINTS: int = Field(default=2, ge=0, le=5)
    TUTOR_MAX_STEPS: int = Field(default=4, ge=1, le=10, description="Questions per session")
    TUTOR_EXPLAIN_MAX_WORDS: int = Field(default=220)
    TUTOR_STATE_SOURCE_CHARS: int = Field(
        default=900, description="Per-source text kept in the tutor state for grading"
    )
    READING_CHUNKS_PER_SECTION: int = Field(default=4, ge=1, le=10)
    SYNTH_TOP_K: int = Field(default=8, ge=2, le=20)
    SYNTH_MAX_PER_PAPER: int = Field(default=2, ge=1)
    ADVISOR_CANDIDATES: int = Field(default=10, ge=3, le=30)
    ADVISOR_MAX_RECOMMENDATIONS: int = Field(default=5, ge=1, le=10)

    # Guardrail (Phase 6)
    GUARDRAIL_ENABLED: bool = Field(default=True, description="Post-answer grounding checks")
    GUARDRAIL_DOMAIN_GATE: bool = Field(
        default=True, description="Refuse clearly non-ML questions without an LLM call"
    )
    GUARDRAIL_MODERATION: bool = Field(
        default=False, description="LLM moderation of inputs (costs one call per message)"
    )
    GUARDRAIL_MIN_SUPPORT: float = Field(
        default=0.25, ge=0, le=1, description="Min word overlap of a cited claim with its source"
    )
    GUARDRAIL_MAX_UNCITED: float = Field(
        default=0.5, ge=0, le=1, description="Max share of substantive sentences without [n]"
    )

    # Observability
    LANGSMITH_TRACING: bool = Field(default=False, description="Opt-in: sends traces to a cloud")
    METRICS_ENABLED: bool = True
    METRICS_TOKEN: str | None = Field(
        default=None, description="If set, /metrics requires 'Authorization: Bearer <token>'"
    )

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
        if v and v.startswith("postgres://"):  # Heroku/Neon style
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        return v


settings = Settings()
