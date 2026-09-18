from qdrant_client import QdrantClient, models 
from qdrant_client.http.exceptions import UnexpectedResponse
from src.pipeline.models import ProcessedPaper
from src.storage.embedder import EmbeddingService 
import numpy as np 
from rich.console import Console 


console = Console()

class VectorStore:
    COLLECTION_NAME = "arxiv_papers_v1"
    
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url = url)
        self.embedder = EmbeddingService()
        self._ensure_collection()
        
    
    def _ensure_collection(self):
        """ 
            create collection if doesn't exists
        """
        try:
            self.client.get_collection(self.COLLECTION_NAME)
        except UnexpectedResponse:
            self.client.create_collection(
                collection_name = self.COLLECTION_NAME,
                vectors_config = models.VectorParams(
                    size = 384,
                    distance = models.Distance.COSINE
                )
            )
            console.print("[blue]Created Qdrant collection.[/blue]")
            

    async def upsert_vectors(self, paper_data: ProcessedPaper):
        if not paper_data.chunks:
            return 
        
        #Step1: Embed content 
        texts = [chunk.content for chunk in paper_data.chunks]
        vectors = self.embedder._model(texts)
        
        #Step2: Prepare payloads
        payloads = []
        ids = []
        
        for i, chunk in enumerate(paper_data.chunks):
            ids.append(chunk.chunk_id)
            payloads.append({
                "paper_id": chunk.paper_id,
                "section": chunk.section_title,
                "content": chunk.content,
                "metadata": chunk.metadata 
            })
            
        self.client.upsert(collection_name=self.COLLECTION_NAME, 
                           points = models.Batch(
                               ids = ids,
                               vectors = vectors.tolist(),
                               payloads = payloads
                           ))
        
        console.print(f"[green] Indexed {len(ids)} vectors to Qdrant.[/green]")