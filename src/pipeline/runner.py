import asyncio 
import os 
from pathlib import Path 
from rich.console import Console 
from src.pipeline.fetcher import ArxivFetcher
from src.pipeline.parser import PDFParser
from src.pipeline.chunker import TextChunker
from src.pipeline.models import ProcessedPaper

from src.storage.database import init_db_schema, AsyncSessionLocal
from src.storage.repositories import PgRepo
from src.storage.vector_db import VectorStore 


console = Console()

    
async def run_pipeline(topic: str, output_dir: str = "data/papers"):
    
    #Initialize Storage Infra
    await init_db_schema()
    vector_store = VectorStore()
    db_repo = PgRepo()
    
    #Initialize Pipeline Components
    fetcher = ArxivFetcher()
    parser = PDFParser()
    chunker = TextChunker()
    
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    
    try:
        #Step 1: Fetch Metadata 
        papers = await fetcher.fetch_papers(
            query = f"all:{topic}",
            max_results = 3
        )
        
        for paper in papers:
            console.rule(f"Processing: {paper.title[:50]}...")
            
            #Step 2: Download the PDF
            pdf_path = os.path.join(output_dir, f"{paper.arxiv_id}.pdf")
            downloaded = await fetcher.download_pdf(paper, pdf_path)
            
            if not downloaded:
                continue 
            
            #Step 3: Parse PDF
            sections = parser.parse(pdf_path)
            
            
            #Step 4: Chunk Content
            chunks = chunker.chunk_sections(paper.arxiv_id, sections)
            
            #Step 5: Assemble 
            processed_paper = ProcessedPaper(metadata = paper,sections = sections,chunks = chunks)
            
            #Step 6: Store
            async with AsyncSessionLocal() as session:
                await db_repo.upsert_paper_and_chunks(session, processed_paper)
                
            await vector_store.upsert_vectors(processed_paper)
            
            console.print(f"[green]Processed {len(chunks)} chunks. [/green]")
            
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        raise 
    

if __name__ == "__main__":
    asyncio.run(run_pipeline("Large Language Models"))
    
        