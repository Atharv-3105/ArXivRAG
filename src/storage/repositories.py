from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select 
from src.storage.database import Paper, PaperChunk
from src.pipeline.models import PaperMetadata, ProcessedPaper
from rich.console import Console

console = Console()

class PgRepo:
    async def upsert_paper_and_chunks(self, session: AsyncSession, paper_data: ProcessedPaper):
        """ 
            Function checks if paper exists? If not, insert's paper + chunks Else skip
        """
        
        paper_meta = paper_data.metadata 
        
        #Check exists?
        result = await session.execute(select(Paper).where(Paper.arxiv_id == paper_meta.arxiv_id))
        existing_paper = result.scalars().first()
        
        if existing_paper:
            console.print(f"[yellow] Paper {paper_meta.arxiv_id} already exists. Skipping[/yellow]")
            return 
        
        #Create Paper Record
        new_paper = Paper(
            arxiv_id = paper_meta.arxiv_id,
            title = paper_meta.title, 
            abstract = paper_meta.abstract,
            authors = paper_meta.authors,
            published_date = paper_meta.published,
            pdf_url = paper_meta.pdf_url,
            categories = paper_meta.categories,
            chunks = [
                PaperChunk(
                    chunk_id = c.chunk_id,
                    section_title = c.section_title,
                    content = c.content,
                    metadata = c.metadata
                ) for c in paper_data.chunks 
            ]
        )
        
        session.add(new_paper)
        await session.commit()
        console.print(f"[green] Saved {paper_meta.arxiv_id} to DB ({len(paper_data.chunks)} chunks). [/green]")
        