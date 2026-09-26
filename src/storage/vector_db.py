"""Qdrant: hybrid (dense + BM25) index over chunks. A derived store - see ADR 0002.

Each point has two named vectors:
  "dense" - bge-small, cosine: matches meaning ("attention cost" ~ "quadratic complexity")
  "bm25"  - sparse term weights, IDF applied by Qdrant: matches exact tokens (model names,
            symbols, acronyms such as "LoRA" or "KV-cache") that dense models blur
and the two ranked lists are fused server-side with Reciprocal Rank Fusion (ADR 0005).
"""

import asyncio
from dataclasses import dataclass
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.core.config import settings
from src.core.exceptions import IndexError, RetrievalError
from src.pipeline.models import ProcessedPaper
from src.storage.embedder import EmbeddingService

logger = structlog.get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"
# Fields we filter on get a payload index; without one Qdrant scans every payload.
INDEXED_FIELDS = ("paper_id", "metadata.math_fidelity")


@dataclass
class SearchHit:
    chunk_id: str
    paper_id: str
    title: str
    section: str
    content: str
    score: float
    metadata: dict
    published: str | None = None  # ISO date, from the payload


class VectorStore:
    def __init__(self):
        self.collection = settings.QDRANT_COLLECTION
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=30
        )
        self.embedder = EmbeddingService()

    async def initialize(self) -> None:
        if await self.client.collection_exists(self.collection):
            await self._check_schema()
            logger.info("Qdrant collection already exists", collection=self.collection)
            return

        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=settings.EMBEDDING_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=2, indexing_threshold=20000
            ),
        )
        for field in INDEXED_FIELDS:
            await self.client.create_payload_index(
                self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )
        logger.info("Created Qdrant collection", collection=self.collection)

    async def _check_schema(self) -> None:
        """Refuse to write into a collection with a different layout (e.g. the old dense-only v1).

        Silently mixing layouts fails later and far away (upserts rejected mid-run, or searches
        that return nothing). Failing at startup with the fix in the message is cheaper.
        """
        info = await self.client.get_collection(self.collection)
        params = info.config.params
        dense = params.vectors.get(DENSE) if isinstance(params.vectors, dict) else None
        sparse = (params.sparse_vectors or {}).get(SPARSE)
        if dense is None or sparse is None or dense.size != settings.EMBEDDING_DIM:
            raise IndexError(
                f"Qdrant collection '{self.collection}' does not have the expected layout "
                f"({DENSE}: {settings.EMBEDDING_DIM}-d dense + {SPARSE}: sparse). Point "
                "QDRANT_COLLECTION at a new name and run `python -m src.pipeline.reindex --all`.",
                code="VECTOR_SCHEMA_MISMATCH",
            )

    async def upsert_vectors(self, paper_data: ProcessedPaper) -> None:
        if not paper_data.chunks:
            logger.warning("No chunks to index", arxiv_id=paper_data.metadata.arxiv_id)
            return

        logger.info("Encoding vectors", chunk_count=len(paper_data.chunks))
        # The section path is embedded with the text so a chunk "knows" where it lives.
        texts = [chunk.embedding_text for chunk in paper_data.chunks]
        # Embedding is CPU-bound; keep it off the event loop.
        dense = await asyncio.to_thread(self.embedder.encode, texts)
        sparse = await asyncio.to_thread(self.embedder.encode_sparse, texts)

        if len(dense) != len(texts) or len(sparse) != len(texts):
            raise IndexError("Embedding count mismatch", code="VECTOR_DIM_ERROR")

        # chunk_id is a deterministic UUID (see TextChunker), so re-ingesting overwrites points
        # instead of duplicating them.
        ids: list[int | str | UUID] = [chunk.chunk_id for chunk in paper_data.chunks]
        meta = paper_data.metadata
        payloads = [
            {
                "paper_id": chunk.paper_id,
                # Denormalised so a search result can be shown and cited without a Postgres read.
                "title": meta.title,
                "published": meta.published.date().isoformat(),
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
        vectors = {
            DENSE: dense.tolist(),
            SPARSE: [models.SparseVector(indices=s.indices, values=s.values) for s in sparse],
        }

        try:
            await self.client.upsert(
                collection_name=self.collection,
                points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
                wait=True,
            )
            logger.info("Vectors upserted", count=len(ids))
        except Exception as e:
            raise IndexError(f"Qdrant upsert failed: {e}", code="VECTOR_UPSERT_ERROR") from e

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        """Hybrid search: dense and BM25 candidates, fused with RRF in one Qdrant round trip."""
        top_k = top_k or settings.RETRIEVAL_TOP_K
        dense = await asyncio.to_thread(self.embedder.encode_query, query)
        sparse = await asyncio.to_thread(self.embedder.encode_sparse_query, query)
        flt = None
        if paper_ids:
            flt = models.Filter(
                must=[models.FieldCondition(key="paper_id", match=models.MatchAny(any=paper_ids))]
            )
        prefetch = [
            models.Prefetch(
                query=dense.tolist(), using=DENSE, limit=settings.RETRIEVAL_PREFETCH, filter=flt
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE,
                limit=settings.RETRIEVAL_PREFETCH,
                filter=flt,
            ),
        ]
        try:
            response = await self.client.query_points(
                collection_name=self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                timeout=int(settings.DEPENDENCY_TIMEOUT_SECONDS) or 1,
            )
        except Exception as e:
            raise RetrievalError(f"Qdrant search failed: {e}", code="VECTOR_SEARCH_ERROR") from e

        hits = []
        for point in response.points:
            p = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(point.id),
                    paper_id=p.get("paper_id", ""),
                    title=p.get("title", ""),
                    section=p.get("section", ""),
                    content=p.get("content", ""),
                    score=point.score,
                    metadata=p.get("metadata") or {},
                    published=p.get("published"),
                )
            )
        return hits

    async def count(self) -> int:
        return (await self.client.count(self.collection, exact=True)).count

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
