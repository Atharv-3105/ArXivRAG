from pydantic import BaseModel, Field 
from typing import List, Optional 
from datetime import datetime 

class PaperMetadata(BaseModel):
    arxiv_id:   str = Field(..., description = "e.g.. 2401.12345")
    title:      str
    authors:    List[str]
    abstract:   str 
    published:  datetime 
    updated:    datetime 
    categories: List[str]
    pdf_url:    str 
    primary_category:   str 
    

class SectionContent(BaseModel):
    title: Optional[str] = None 
    text:   str 
    page_start:    int 
    page_end:      int 
    

class Chunk(BaseModel):
    chunk_id:       str 
    paper_id:       str 
    section_title:  str 
    content:        str 
    metadata:       dict = Field(default_factory=dict)
    
class ProcessedPaper(BaseModel):
    metadata:   PaperMetadata
    sections:   List[SectionContent]
    chunks:     List[Chunk]
    
    