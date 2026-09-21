import asyncio
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.core.config import settings
from src.core.exceptions import IndexError
from src.pipeline.models import ProcessedPaper
from src.storage.embedder import EmbeddingService

logger = structlog.get_logger(__name__)


class VectorStore:
    def __init__(self):
        self.collection = settings.QDRANT_COLLECTION
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=30
        )
        self.embedder = EmbeddingService()

    async def initialize(self) -> None:
        if await self.client.collection_exists(self.collection):
            logger.info("Qdrant collection already exists", collection=self.collection)
            return

        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIM, distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=2, indexing_threshold=20000
            ),
        )
        logger.info("Created Qdrant collection", collection=self.collection)

    async def upsert_vectors(self, paper_data: ProcessedPaper) -> None:
        if not paper_data.chunks:
            logger.warning("No chunks to index", arxiv_id=paper_data.metadata.arxiv_id)
            return

        logger.info("Encoding vectors", chunk_count=len(paper_data.chunks))
        # The section path is embedded with the text so a chunk "knows" where it lives.
        texts = [chunk.embedding_text for chunk in paper_data.chunks]
        # Embedding is CPU-bound; keep it off the event loop.
        vectors = await asyncio.to_thread(self.embedder.encode, texts)

        if len(vectors) != len(texts):
            raise IndexError("Embedding count mismatch", code="VECTOR_DIM_ERROR")

        # chunk_id is a deterministic UUID (see TextChunker), so re-ingesting overwrites points
        # instead of duplicating them.
        ids: list[int | str | UUID] = [chunk.chunk_id for chunk in paper_data.chunks]
        payloads = [
            {
                "paper_id": chunk.paper_id,
                "section": chunk.section_title,
                "content": chunk.content,
                "metadata": {
                    **chunk.metadata,
                    "parse_method": paper_data.parse_method,
                    "math_fidelity": paper_data.math_fidelity,
                },
            }
            for chunk in paper_data.chunks
        ]

        try:
            await self.client.upsert(
                collection_name=self.collection,
                points=models.Batch(ids=ids, vectors=vectors.tolist(), payloads=payloads),
                wait=True,
            )
            logger.info("Vectors upserted", count=len(ids))
        except Exception as e:
            raise IndexError(f"Qdrant upsert failed: {e}", code="VECTOR_UPSERT_ERROR") from e

    async def delete_paper(self, arxiv_id: str) -> None:
        """Remove every vector of one paper (used when re-ingesting it with a better parse)."""
        try:
            await self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="paper_id", match=models.MatchValue(value=arxiv_id)
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as e:
            raise IndexError(f"Qdrant delete failed: {e}", code="VECTOR_DELETE_ERROR") from e

    async def close(self) -> None:
        await self.client.close()
