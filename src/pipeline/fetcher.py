import httpx 
import asyncio 
import xml.etree.ElementTree as ET 
from datetime import datetime 
from typing import List 
import structlog
from src.pipeline.models import PaperMetadata
from src.core.exceptions import FetchError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

logger = structlog.get_logger(__name__)

class ArxivFetcher:
    BASE_URL = "http://export.arxiv.org/api/query"
    RATE_LIMIT_DELAY = 1.2 
    
    def __init__(self):
        self._semaphore = asyncio.Semaphore(1)  #TO enforce sequential requests
        self._client = httpx.AsyncClient(timeout = 30.0, follow_redirects=True)
        
    @retry(stop = stop_after_attempt(3), wait = wait_exponential_jitter(initial = 1.0, max = 10.0),
           retry = retry_if_exception_type((httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException)),
           reraise = True)
    
    async def fetch_papers(self, query: str, max_results: int = 10, sort_by: str = "submittedDate", sort_order: str = "descending") -> List[PaperMetadata]:
        """ 
            Function to fetch papers from ArXiv while maintaining Rate-Limits and Retry Logic
        """
        async with self._semaphore:
            logger.info("Fetching papers from ArXiv", query = query, max_results = max_results)
            params = {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order
            }
            
            try:
                response = await self._client.get(self.BASE_URL, params=params)
                response.raise_for_status()
            except Exception as e:
                raise FetchError(f"ArXiv API request failed: {e}", code = "FETCH_API_ERROR", metadata = {"query": query})
                    
            root = ET.fromstring(response.text)
            
            #Namespace for Atom feed
            ns = {'atom': "http://www.w3.org/2005/Atom"}
            
            papers = []
            for entry in root.findall('atom:entry', ns):
                try:
                    papers.append(self._parse_entry(entry, ns))
                except Exception as e:
                    logger.warning("Failed to parse ArXiv entry", error = str(e))
                    continue 
            
            logger.info("Successfully parsed paper", count = len(papers))
            await asyncio.sleep(self.RATE_LIMIT_DELAY)
            return papers 
                        
    def _parse_entry(self, entry, ns: dict) -> PaperMetadata:
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        published = entry.find('atom:published', ns).text
        updated = entry.find('atom:updated', ns).text 
        
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        categories = [c.get('term')for c in entry.findall('atom:category', ns)]
        
        #Get the PDF Link of the Paper
        pdf_link = None 
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_link = link.get('href')
                
        
        #Extract the ArXiv Paper ID from URI
        id_uri = entry.find('atom:id', ns).text 
        arxiv_id = id_uri.split('/')[-1]
        
        return PaperMetadata(
            arxiv_id = arxiv_id,
            title =title,
            authors = authors,
            abstract = summary,
            published = datetime.fromisoformat(published),
            updated = datetime.fromisoformat(updated),
            categories = categories,
            pdf_url = pdf_link,
            primary_category = categories[0] if categories else "Unknown"
        )
        

    async def download_pdf(self, paper: PaperMetadata, save_path: str) -> bool:
        """ 
            Function to download Pdfs asynchronously
        """
        logger.info("Downloading PDF", arxiv_id = paper.arxiv_id)
        try:
            async with self._client.stream("GET", paper.pdf_url) as resp:
                resp.raise_for_status()
                with open(save_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
            
            logger.info("PDF downloaded successfully", arxiv_id = paper.arxiv_id, path = save_path)                
            return True 

        except Exception as e:
            raise FetchError(f"PDF Download failed: {e}", code = "FETCH_PDF_ERROR", metadata = {"arxiv_id": paper.arxiv_id})
        