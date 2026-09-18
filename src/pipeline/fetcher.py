import httpx 
import asyncio 
import xml.etree.ElementTree as ET 
from datetime import datetime 
from typing import List 
from rich.console import Console 
from src.pipeline.models import PaperMetadata

console = Console()

class ArxivFetcher:
    BASE_URL = "http://export.arxiv.org/api/query"
    
    async def fetch_papers(self, query: str, max_results: int = 10, sort_by: str = "submittedDate", sort_order: str = "descending") -> List[PaperMetadata]:
        """ 
            Function to fetch papers from ArXiv while maintaining Rate-Limits and Retry Logic
        """
        
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        console.print(f"Fetching papers for query: {query}")
        
        try:
            #ArXiv rate-limit is 3 req/s so We wait for 1.5s between each request
            async with httpx.AsyncClient(timeout = 30.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.text)
                
                #Namespace for Atom feed
                ns = {'atom': "http://www.w3.org/2005/Atom"}
                
                papers = []
                for entry in root.findall('atom:entry', ns):
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
                    
                    papers.append(PaperMetadata(
                        arxiv_id = arxiv_id,
                        title = title,
                        authors = authors,
                        abstract = summary,
                        published = datetime.fromisoformat(published),
                        updated = datetime.fromisoformat(updated),
                        categories = categories,
                        pdf_url = pdf_link,
                        primary_category = categories[0] if categories else "Unknown"
                    ))
                    
                console.print(f"[green]Successfully fetched {len(papers)} papers.[/green]")
                return papers 
        except httpx.HTTPError as e:
            console.print(f"[red]Failed to fetch papers: {e}[/red]")
            raise 
        

    async def download_pdf(self, paper: PaperMetadata, save_path: str) -> bool:
        """ 
            Function to download Pdfs asynchronously
        """
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", paper.pdf_url) as responses:
                    responses.raise_for_status()
                    with open(save_path, "wb") as f:
                        async for chunk in responses.aiter_bytes():
                            f.write(chunk)
                            
            console.print(f"[blue]Downloaded PDF for {paper.arxiv_id}[/blue]")
            return True 

        except Exception as e:
            console.print(f"[red]Failed to download {paper.arxiv_id}: {e}[/red]")
            return False 