import fitz
import re 
from typing import List
from src.pipeline.models import SectionContent

class PDFParser:
    #Regex for common section headers like "1.Introduction, 2.Method etc"
    SECTION_PATTERN = re.compile(r"^\d+\.\s+([A-Za-z\s]+)", re.MULTILINE)
    
    def parse(self, pdf_path: str)-> List[SectionContent]:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        #Extract text from all pages 
        for page in doc:
            full_text += page.get_text()
            
        
        sections = self._split_into_sections(full_text)
        doc.close()
        return sections 
    
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