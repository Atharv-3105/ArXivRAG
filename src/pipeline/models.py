from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ParseMethod = Literal["html", "pdf"]
# "latex": math is real LaTeX taken from the source markup. "degraded": recovered from PDF glyphs,
# so equations may be garbled and must not be treated as authoritative.
MathFidelity = Literal["latex", "degraded"]


class PaperMetadata(BaseModel):
    arxiv_id: str = Field(..., description="e.g. 2401.12345v1")
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    categories: list[str]
    pdf_url: str
    primary_category: str


class Equation(BaseModel):
    """A display equation. Inline math stays in the section text as `$...$`."""

    ordinal: int
    latex: str
    number: str | None = None  # the label the author typeset, e.g. "(3)"
    display: bool = True
    section_path: str = ""
    anchor: str | None = None  # element id in the arXiv HTML, e.g. "S2.E3"


class Reference(BaseModel):
    ordinal: int
    key: str | None = None
    text: str
    urls: list[str] = Field(default_factory=list)


class CitationContext(BaseModel):
    """One in-text citation: the sentence around it and the bibliography keys it points to.
    HTML tier only (PDF text has no link from "[12]" to a bibliography entry)."""

    keys: list[str]  # Reference.key values, e.g. ["bib.bib12", "bib.bib13"]
    sentence: str
    anchor: str | None = None  # id of the enclosing section


class SectionContent(BaseModel):
    title: str | None = None
    text: str
    page_start: int | None = None  # PDF tier only
    page_end: int | None = None
    level: int = 1  # 1 = section, 2 = subsection, ...
    path: str = ""  # "3 Method > 3.1 Setup"
    anchor: str | None = None  # HTML tier: id usable as arxiv.org/html/<id>#<anchor>


class ParsedDocument(BaseModel):
    """What a parser tier hands to the rest of the pipeline."""

    method: ParseMethod
    math_fidelity: MathFidelity
    sections: list[SectionContent]
    equations: list[Equation] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    citation_contexts: list[CitationContext] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    section_title: str
    content: str
    context: str = ""  # section path; prepended when embedding so a chunk knows where it lives
    metadata: dict = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        return f"{self.context}\n{self.content}" if self.context else self.content


class ProcessedPaper(BaseModel):
    metadata: PaperMetadata
    sections: list[SectionContent]
    chunks: list[Chunk]
    equations: list[Equation] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    citation_contexts: list[CitationContext] = Field(default_factory=list)
    parse_method: ParseMethod = "pdf"
    math_fidelity: MathFidelity = "degraded"
    parse_stats: dict = Field(default_factory=dict)
