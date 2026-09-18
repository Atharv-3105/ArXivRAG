from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models 

from src.pipeline.models import ProcessedPaper
from src.storage.embedder import EmbeddingService 
from src.core.exceptions import IndexError
import structlog 

logger = structlog.get_logger(__name__)

class VectorStore:
    COLLECTION_NAME = "arxiv_papers_v1"
    
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = AsyncQdrantClient(url = url, timeout = 30)
        self.embedder = EmbeddingService()
    
    async def initialize(self):
        try:
            await self.client.get_collection(self.COLLECTION_NAME)
            logger.info("Qdrant collection already exists")
            
        except UnexpectedResponse:
            await self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config = models.VectorParams(size = 384, distance = models.Distance.COSINE),
                optimizers_config = models.OptimizersConfigDiff(default_segment_number=2, indexing_threshold=20000),
            )
            
            logger.info("Created Qdrant collection")
            
    
    async def upsert_vectors(self, paper_data: ProcessedPaper) -> None:
        if not paper_data.chunks:
            logger.warning("No chunks to index", arxiv_id = paper_data.metadata.arxiv_id)
            return 
        
        logger.info("Encoding vectors", chunk_count = len(paper_data.chunks))
        texts = [chunk.content for chunk in paper_data.chunks]
        vectors = self.embedder.encode(texts)
        
        if len(vectors) != len(texts):
            raise IndexError("Embedding dimension mismatch", code = "VECTOR_DIM_ERROR")
        
        ids = [chunk.chunk_id for chunk in paper_data.chunks]
        payloads = [
            {
                "paper_id": chunk.paper_id,
                "section": chunk.section_title,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in paper_data.chunks 
        ]        
        
        try:
            await self.client.upsert(collection_name= self.COLLECTION_NAME,
                                     points = models.Batch(ids = ids, vectors = vectors.tolist(), payloads = payloads),
                                     wait = True)
            logger.info("vectors upserted successfully", count = len(ids))
        
        except Exception as e:
            raise IndexError(f"Qdrant upsert failed: {e}", code = "VECTOR_UPSERT_ERROR")