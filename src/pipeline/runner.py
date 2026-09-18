import asyncio 
import os 
from pathlib import Path 
import structlog 

from src.pipeline.fetcher import ArxivFetcher
from src.pipeline.parser import PDFParser
from src.pipeline.chunker import TextChunker
from src.pipeline.models import ProcessedPaper

from src.storage.database import init_db_schema, get_async_session, FailedPaper
from src.storage.repositories import PgRepo
from src.storage.vector_db import VectorStore 

from src.core.exceptions import ArxivMindError, ParseError, FetchError, IndexError

logger = structlog.get_logger(__name__)


@dataclass 
class PipelineMetrics:
    fetched: int = 0
    parsed: int = 0
    indexed: int = 0
    failed: int = 0
    skipped: int = 0

async def run_pipeline(topic: str, output_dir: str = "data/papers", max_papers: int = 5) -> PipelineMetrics:
    
    metrics = PipelineMetrics()

    #Initialize Storage Infra
    await init_db_schema()
    vector_store = VectorStore()
    await vector_store.initialize()
    db_repo = PgRepo()
    
    #Initialize Pipeline Components
    fetcher = ArxivFetcher()
    parser = PDFParser()
    chunker = TextChunker()
    
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    
    logger.info("Pipeline started", topic = topic, max_papers = max_papers)
    
    try:
        #Step 1: Fetch Metadata 
        papers = await fetcher.fetch_papers(
            query = f"all:{topic}",
            max_results = max_papers,
        )
        metrics.fetched = len(papers)
        
        for paper in papers:
            try:
                logger.info("processing paper", arxiv_id = paper.arxiv_id)
                pdf_path = Path(output_dir) / f"{paper.arxiv_id}.pdf"
                
                if not pdf_path.exists():
                    await fetcher.download_pdf(paper, str(pdf_path))
                    
                sections, parse_metrics = parser.parse(str(pdf_path))
                metrics.parsed += 1
                
                chunks = chunker.chunk_sections(paper.arxiv_id, sections)
                if len(chunks) < 2:
                    logger.warning("Skipping: insufficient chunks", arxiv_id = paper.arxiv_id)
                    metrics.skipped += 1
                    continue 
                
                processed = ProcessedPaper(metadata = paper, sections = sections, chunks = chunks)
                
                #Atomic Transaction: PG + Qdrant
                async for session in get_async_session:
                    inserted = await db_repo.upsert_paper_and_chunks(session, processed)
                    if inserted:
                        await vector_store.upsert_vectors(processed)
                        metrics.indexed += len(chunks)
                        
                logger.info("Successfully processed", arxiv_id = paper.arxiv_id)
                
            except (FetchError, ParseError, IndexError) as e:
                metrics.failed += 1
                
                #If Upserting failed add the Paper to DLQ
                e.log_error(arxiv_id = paper.arxiv_id)
                async for session in get_async_session():
                    await db_repo.log_failure_to_dlq(session, paper.arxiv_id, str(e))
                    
            except Exception as e:
                metrics.failed += 1
                logger.error("Unexpected pipeline error", arxiv_id = paper.arxiv_id, error = str(e))
                
                #If any other failure reason then also add to DLQ
                async for session in get_async_session():
                    await db_repo.log_failure_to_dlq(session, paper.arxiv_id, f"UNKNOWN: {e}")
    except Exception as e:
        logger.error("Pipeline execution failed", error = str(e))
        raise 
    
    finally:
        logger.info("Pipeline completed", metrics = vars(metrics))
        return metrics 
    

if __name__ == "__main__":
    asyncio.run(run_pipeline("Large Language Models", max_papers = 3))