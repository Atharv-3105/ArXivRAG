import fitz
import re 
from typing import List, Tuple
import structlog 
from src.pipeline.models import SectionContent
from src.core.exceptions import ParseError

logger = structlog.get_logger(__name__)

#Regex for common section headers like "1.Introduction, 2.Method etc"
SECTION_PATTERN = re.compile(r"^\d+\.\s+([A-Za-z\s]+)", re.MULTILINE)


class ParseMetrics:
    def __init__(self):
        self.total_pages: int = 0
        self.total_words: int = 0
        self.section_count: int = 0
        self.latex_block_count: int = 0
        self.is_valid: bool = False 

class PDFParser:
    MIN_WORD_THRESHOLD = 800
    MIN_SECTION_THRESHOLD = 2
    
    def parse(self, pdf_path: str)-> Tuple[List[SectionContent], ParseMetrics]:
        logger.info("parsing pdf", path = pdf_path)
        doc = fitz.open(pdf_path)
        metrics = ParseMetrics()
        metrics.total_pages = len(doc)
        
        full_text = ""
        
        #Extract text from all pages 
        for page in doc:
            full_text += page.get_text()
            metrics.total_words += len(page.get_text().split())
            #Get the count of blocks in paper which are LaTeX
            metrics.latex_block_count += full_text.count(r"\begin{")
            
        doc.close()
        sections = self._split_into_sections(full_text)
        metrics.section_count = len(sections)
        
        #Quality Checks
        if metrics.total_words < self.MIN_WORD_THRESHOLD:
            raise ParseError(f"Paper too short: {metrics.total_words} words", code = "PARSE_LENGTH_ERROR")
        if metrics.section_count < self.MIN_SECTION_THRESHOLD:
            raise ParseError(f"Too few sections detected: {metrics.section_count}", code = "PARSE_STRUCTURE_ERROR")
        
        metrics.is_valid = True 
        logger.info("Parse validation passed", metrics = vars(metrics))
        return sections , metrics 
    
    def _split_into_sections(self, text: str) -> List[SectionContent]:
        lines = text.split('\n')
        sections = []
        current_section = "Abstract"
        current_text = []
        
        for line in lines:
            match = self.SECTION_PATTERN.match(line.strip())
            if match:
                #Save previous section
                if current_text:
                    sections.append(SectionContent(title = current_section, text = "\n".join(current_text).strip()))
                    current_section = match.group(1).strip()
                    current_text = [line]
                    
                else:
                    current_text.append(line)
                    
        
        #Append the lasst section
        if current_text:
            sections.append(SectionContent(title = current_section, text = "\n".join(current_text).strip()))
            
            
        return sections 