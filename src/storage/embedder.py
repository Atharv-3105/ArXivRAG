"""Dense + sparse text embeddings via fastembed (ONNX Runtime).

fastembed instead of sentence-transformers: the same bge-small weights exported to ONNX, without
PyTorch, by far the largest dependency, which is what lets the API fit a free host. The BM25
sparse encoder for hybrid search comes from the same library.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from src.core.config import settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class EmbeddingService:
    """Lazy-loading: models are downloaded/loaded on the first real encode, never at import."""

    def __init__(
        self,
        model_name: str | None = None,
        sparse_model_name: str | None = None,
        batch_size: int | None = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.sparse_model_name = sparse_model_name or settings.SPARSE_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model: Any = None  # fastembed.TextEmbedding (imported lazily)
        self._sparse: Any = None  # fastembed.SparseTextEmbedding

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding  # heavy import (onnxruntime): keep it lazy

            logger.info("Loading embedding model", model=self.model_name)
            self._model = TextEmbedding(self.model_name, cache_dir=settings.MODEL_CACHE_DIR)
        return self._model

    @property
    def sparse_model(self):
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            logger.info("Loading sparse model", model=self.sparse_model_name)
            self._sparse = SparseTextEmbedding(
                self.sparse_model_name, cache_dir=settings.MODEL_CACHE_DIR
            )
        return self._sparse

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Documents -> L2-normalised dense vectors. CPU-bound: call via asyncio.to_thread."""
        if not texts:
            return np.array([])
        vectors = np.array(list(self.model.embed(texts, batch_size=self.batch_size)))
        return self._normalise(vectors)

    def encode_query(self, query: str) -> np.ndarray:
        """Short query -> dense vector. bge recommends an instruction prefix for short queries
        against long passages; it is config (EMBEDDING_QUERY_PREFIX) so the eval can measure it."""
        text = f"{settings.EMBEDDING_QUERY_PREFIX}{query}"
        return self._normalise(np.array(list(self.model.embed([text]))))[0]

    def encode_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Documents -> BM25 term weights (term-frequency part; Qdrant applies IDF at query time)"""
        if not texts:
            return []
        return [
            SparseVector(e.indices.tolist(), e.values.tolist())
            for e in self.sparse_model.embed(texts, batch_size=self.batch_size)
        ]

    def encode_sparse_query(self, query: str) -> SparseVector:
        e = next(iter(self.sparse_model.query_embed(query)))
        return SparseVector(e.indices.tolist(), e.values.tolist())
