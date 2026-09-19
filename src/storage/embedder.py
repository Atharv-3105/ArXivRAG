import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from src.core.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Lazy-loading wrapper: the model (~130MB for bge-small) loads on first real encode."""

    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Convert text chunks to L2-normalised vectors. CPU-bound: call via asyncio.to_thread."""
        if not texts:
            return np.array([])
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
