from sentence_transformers import SentenceTransformer 
from typing import List 
import numpy as np 

class EmbeddingService:
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    
    def __init__(self):
        self._model = None 
        
    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """ 
            converts list of text chunks to list of vectors
        """
        if not texts:
            return np.array([])
    
        return self.model.encode(texts, normalize_embeddings = True)
    
    