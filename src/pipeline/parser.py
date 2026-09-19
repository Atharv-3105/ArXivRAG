import re

import pymupdf
import structlog

from src.core.config import settings
from src.core.exceptions import ParseError
from src.pipeline.models import SectionContent

logger = structlog.get_logger(__name__)

FRONT_MATTER = "Front Matter"
DROPPED_SECTIONS = {"references", "bibliography"}  # citation lists are noise for retrieval

# Unnumbered headings we recognise by name (deliberately excludes ambiguous words such as
# "Model" or "Analysis", which also appear as figure/table labels).
_KNOWN = (
    r"abstract|introduction|related work|background|preliminaries|methods?|methodology|"
    r"experiments?|experimental setup|results(?: and discussion)?|discussion|"
    r"conclusions?(?: and future work)?|future work|limitations|acknowledge?ments?|"
    r"references|bibliography|appendix(?: [a-z])?"
)
KNOWN_HEADING = re.compile(rf"^(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?({_KNOWN})\.?$", re.IGNORECASE)
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")  # "2 Related Work", "2.1. Setup"
ROMAN_HEADING = re.compile(r"^([IVX]+)\.\s+(\S.*)$")  # "II. THEORY"
NUMBER_ONLY = re.compile(r"^(\d+(?:\.\d+)*)\.?$")  # "1" ... title is on the next line

_STOPWORDS = {"a", "an", "the", "of", "and", "or", "for", "in", "on", "to", "with", "via", "by"}
_STOPWORDS |= {"from", "at", "as", "is", "are", "vs"}
_TITLE_CHARS = re.compile(r"^[A-Za-z][A-Za-z0-9 \-:,&/()'’]*$")


def _looks_like_title(text: str) -> bool:
    """Short, no sentence punctuation, and mostly capitalised words ("Related Work" yes,
    "Models that are generally used" no)."""
    text = text.strip()
    if not 2 <= len(text) <= 70 or not _TITLE_CHARS.match(text):
        return False
    words = text.split()
    if len(words) > 9:
        return False
    significant = [w for w in words if w.lower() not in _STOPWORDS]
    if not significant:
        return False
    capitalised = sum(1 for w in significant if w[0].isupper())
    return capitalised / len(significant) >= 0.6


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_text(text: str) -> str:
    """Drop control characters (keeps \\n and \\t). PDF extraction can emit NUL bytes, which
    Postgres TEXT columns reject outright ("invalid byte sequence for encoding UTF8: 0x00")."""
    return _CONTROL_CHARS.sub("", text)


def _tidy_title(title: str) -> str:
    title = " ".join(title.split()).rstrip(".:")
    return title.title() if title.isupper() else title


class ParseMetrics:
    def __init__(self):
        self.total_pages: int = 0
        self.total_words: int = 0
        self.section_count: int = 0
        self.headed_section_count: int = 0
        self.latex_block_count: int = 0
        self.is_valid: bool = False


class PDFParser:
    """Text extraction + heuristic section splitting (PyMuPDF).

    Known limits: two-column layouts can interleave, and heading detection is heuristic. Papers
    whose structure cannot be recovered are rejected (ParseError) and land in the DLQ instead of
    polluting the index.
    """

    def __init__(self, min_words: int | None = None, min_sections: int | None = None):
        self.min_words = min_words or settings.PARSE_MIN_WORDS
        self.min_sections = min_sections or settings.PARSE_MIN_SECTIONS

    def parse(self, pdf_path: str) -> tuple[list[SectionContent], ParseMetrics]:
        logger.info("Parsing PDF", path=pdf_path)
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            raise ParseError(f"Cannot open PDF: {e}", code="PARSE_OPEN_ERROR") from e

        metrics = ParseMetrics()
        lines: list[tuple[str, int]] = []  # (line, 1-based page number)
        try:
            metrics.total_pages = len(doc)
            for page_no, page in enumerate(doc.pages(), start=1):
                text = _clean_text(page.get_text())
                metrics.total_words += len(text.split())
                metrics.latex_block_count += text.count("\\begin{")
                lines.extend((line, page_no) for line in text.split("\n"))
        except Exception as e:
            raise ParseError(f"Failed to extract text: {e}", code="PARSE_EXTRACT_ERROR") from e
        finally:
            doc.close()

        sections = self._split_into_sections(lines)
        metrics.section_count = len(sections)
        metrics.headed_section_count = sum(1 for s in sections if s.title != FRONT_MATTER)

        if metrics.total_words < self.min_words:
            raise ParseError(
                f"Paper too short: {metrics.total_words} words", code="PARSE_LENGTH_ERROR"
            )
        if metrics.headed_section_count < self.min_sections:
            raise ParseError(
                f"Too few sections detected: {metrics.headed_section_count}",
                code="PARSE_STRUCTURE_ERROR",
            )

        metrics.is_valid = True
        logger.info("Parse validation passed", metrics=vars(metrics))
        return sections, metrics

    def _split_into_sections(self, lines: list[tuple[str, int]]) -> list[SectionContent]:
        sections: list[SectionContent] = []
        title = FRONT_MATTER
        body: list[str] = []
        start_page = lines[0][1] if lines else 1
        last_page = start_page
        last_top_number = 0  # numbered headings must run 1, 2, 3... which filters out list items

        def flush() -> None:
            text = "\n".join(body).strip()
            if text and title.lower() not in DROPPED_SECTIONS:
                sections.append(
                    SectionContent(
                        title=title, text=text, page_start=start_page, page_end=last_page
                    )
                )

        i = 0
        while i < len(lines):
            raw, page = lines[i]
            line = raw.strip()
            heading, consumed = self._match_heading(lines, i, last_top_number)
            if heading is not None:
                flush()
                title, body, start_page = heading[0], [], page
                if heading[1] is not None:
                    last_top_number = heading[1]
                i += consumed
            else:
                body.append(raw)
                i += 1
            if line:
                last_page = page

        flush()
        return sections

    def _match_heading(
        self, lines: list[tuple[str, int]], i: int, last_top: int
    ) -> tuple[tuple[str, int | None] | None, int]:
        """Return ((title, top_level_number), lines_consumed) if lines[i] starts a heading."""
        line = lines[i][0].strip()
        if not line:
            return None, 1

        def in_sequence(number: str) -> bool:
            top = int(number.split(".")[0])
            return last_top <= top <= last_top + 2

        def top_of(number: str) -> int:
            return int(number.split(".")[0])

        if m := KNOWN_HEADING.match(line):
            return (_tidy_title(m.group(1)), None), 1

        if (m := NUMBERED_HEADING.match(line)) and _looks_like_title(m.group(2)):
            if in_sequence(m.group(1)):
                return (_tidy_title(m.group(2)), top_of(m.group(1))), 1

        if (m := ROMAN_HEADING.match(line)) and _looks_like_title(m.group(2)):
            return (_tidy_title(m.group(2)), None), 1

        if (m := NUMBER_ONLY.match(line)) and in_sequence(m.group(1)):
            # Real arXiv PDFs often put the number and the title on separate lines.
            nxt = i + 1
            while nxt < len(lines) and not lines[nxt][0].strip():
                nxt += 1
            if nxt < len(lines):
                candidate = lines[nxt][0].strip()
                if _looks_like_title(candidate) and not NUMBER_ONLY.match(candidate):
                    return (_tidy_title(candidate), top_of(m.group(1))), nxt - i + 1

        return None, 1
