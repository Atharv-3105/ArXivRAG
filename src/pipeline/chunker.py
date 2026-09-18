import uuid 
from typing import List
from src.pipeline.models import Chunk, SectionContent 

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap
        
    
    def chunk_sections(self, paper_id: str, sections: List[SectionContent]) -> List[Chunk]:
        all_chunks = []
        
        for section in sections:
            #Skip very short sections
            if len(section.text) < 100:
                continue 
            
            #Split section text into chunks 
            chunks = self._split_text(section.text)
            
            for i, text in enumerate(chunks):
                chunk_id = f"{paper_id}_{section.title}_{i}"
                all_chunks.append(Chunk(
                    chunk_id = chunk_id,
                    paper_id = paper_id,
                    section_title = section.title,
                    content = text, 
                    metadata = {
                        "word_count": len(text.split()),
                        "section_index": i,
                    }
                ))
                
        
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            if len(" ".join(current_chunk)) >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                
                #overlap: Kepp the last N  words
                current_chunk = current_chunk[-self.chunk_overlap:]
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks 
            