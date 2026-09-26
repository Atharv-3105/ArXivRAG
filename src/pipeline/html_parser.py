"""Parser for arXiv's HTML rendering (LaTeXML output, https://arxiv.org/html/<id>).

Why this tier exists: every <math> element carries its original LaTeX in `alttext`, sections are a
real hierarchy with stable ids, and the bibliography is structured. None of that survives in a PDF.

Output text is Markdown-flavoured: inline math as `$...$`, display math as `$$...$$`, so both the
embedding model and the LLM see the actual formulas.
"""

import re

import structlog
from lxml import etree, html

from src.core.config import settings
from src.core.exceptions import ParseError
from src.pipeline.models import (
    CitationContext,
    Equation,
    ParsedDocument,
    Reference,
    SectionContent,
)

logger = structlog.get_logger(__name__)

SECTION_CLASSES = (
    "ltx_section",
    "ltx_subsection",
    "ltx_subsubsection",
    "ltx_paragraph",
    "ltx_appendix",
)
SKIP_CLASSES = {"ltx_rdf", "ltx_graphics", "ltx_bibliography", "ltx_pagination", "ltx_dates"}
SKIP_TAGS = {"script", "style", "svg", "img", "noscript", "button", "nav", "head"}
BLOCK_TAGS = {"p", "div", "table", "figure", "ul", "ol", "section", "pre", "blockquote"}
MAX_TABLE_ROWS = 60
MAIN_TEXT = "Main Text"
MAX_CONTEXT_CHARS = 600
# Sentence boundary: end punctuation, space, then something that can start a sentence.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[$])")


def _classes(el: etree._Element) -> set[str]:
    return set((el.get("class") or "").split())


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _math_text(m: etree._Element) -> str:
    alt = m.get("alttext")
    if alt and alt.strip():
        return alt.strip()
    return _clean("".join(m.itertext()))  # rare: no alttext -> whatever glyphs are there


# `$` delimits math, so a literal dollar sign in prose is written `\$` and never starts a span.
_MATH_SPAN = re.compile(r"((?<!\\)\$\$.+?(?<!\\)\$\$|(?<!\\)\$.+?(?<!\\)\$)", re.DOTALL)


def _esc(text: str) -> str:
    return text.replace("$", r"\$")


def _tidy(text: str) -> str:
    """Fix spacing around punctuation next to math, WITHOUT touching anything inside `$...$`."""
    pieces = _MATH_SPAN.split(text)
    for i in range(0, len(pieces), 2):  # even indices are plain text, odd indices are math spans
        pieces[i] = re.sub(r"\(\s+", "(", re.sub(r"\s+([,.;:)\]])", r"\1", pieces[i]))
    return "".join(pieces)


class HtmlParser:
    """arXiv HTML -> sections + display equations + references. Pure function of the HTML string."""

    def __init__(self, min_words: int | None = None, min_sections: int = 1):
        # The HTML structure is reliable (unlike PDF heading heuristics), so one section is enough.
        self.min_words = min_words or settings.PARSE_MIN_WORDS
        self.min_sections = min_sections

    # ------------------------------------------------------------------ public

    def parse(self, html_text: str) -> ParsedDocument:
        try:
            root = html.fromstring(html_text)
        except (etree.ParserError, etree.XMLSyntaxError, ValueError) as e:
            raise ParseError(f"Unreadable HTML: {e}", code="PARSE_HTML_ERROR") from e

        article = root.xpath('//article[contains(@class,"ltx_document")]')
        if not article:
            raise ParseError("Not an arXiv LaTeXML document", code="PARSE_HTML_ERROR")
        doc = article[0]

        self._equations: list[Equation] = []
        sections: list[SectionContent] = []

        abstract = doc.xpath('.//div[contains(@class,"ltx_abstract")]')
        if abstract:
            text = "\n\n".join(self._blocks(abstract[0], "Abstract"))
            if text:
                sections.append(
                    SectionContent(
                        title="Abstract", text=text, level=1, path="Abstract", anchor="abstract"
                    )
                )

        pending: list[str] = []
        self._walk_body(doc, sections, pending)
        self._flush_main(sections, pending)

        references = self._references(doc)
        contexts = self._citation_contexts(doc)
        stats = self._stats(sections, references)
        stats["citation_contexts"] = len(contexts)

        if stats["words"] < self.min_words:
            raise ParseError(f"Paper too short: {stats['words']} words", code="PARSE_LENGTH_ERROR")
        if stats["sections"] < self.min_sections:
            raise ParseError(
                f"Too few sections detected: {stats['sections']}", code="PARSE_STRUCTURE_ERROR"
            )

        return ParsedDocument(
            method="html",
            math_fidelity="latex",
            sections=sections,
            equations=self._equations,
            references=references,
            citation_contexts=contexts,
            stats=stats,
        )

    # ---------------------------------------------------------------- sections

    def _walk_body(self, el: etree._Element, out: list[SectionContent], pending: list[str]) -> None:
        """Walk the document in order. Content that sits directly in the body, outside any
        <section> (letter-style papers such as PRL), is gathered into an implicit "Main Text"."""
        for child in el:
            if not isinstance(child.tag, str) or child.tag in SKIP_TAGS:
                continue
            cls = _classes(child)
            joined = " ".join(cls)
            if cls & SKIP_CLASSES or "ltx_abstract" in cls or "ltx_authors" in cls:
                continue
            if "ltx_title" in joined or child.tag == "h1":
                continue
            if child.tag == "section" and cls & set(SECTION_CLASSES):
                self._flush_main(out, pending)
                self._walk_section(child, parent_path="", out=out)
            elif (
                child.tag in {"div", "main"}
                and "ltx_para" not in cls
                and child.xpath(
                    './/section[contains(@class,"ltx_section") or contains(@class,"ltx_appendix")]'
                )
            ):
                self._walk_body(child, out, pending)  # unclassed wrapper around sections
            else:
                pending.extend(self._render_block(child, MAIN_TEXT))

    @staticmethod
    def _flush_main(out: list[SectionContent], pending: list[str]) -> None:
        text = "\n\n".join(b for b in pending if b)
        pending.clear()
        if text:
            out.append(SectionContent(title=MAIN_TEXT, text=text, level=1, path=MAIN_TEXT))

    def _child_sections(self, el: etree._Element) -> list[etree._Element]:
        """Sections that belong directly to `el` (they may be wrapped in unclassed containers)."""
        found: list[etree._Element] = []
        for child in el:
            if not isinstance(child.tag, str):
                continue
            if child.tag == "section" and _classes(child) & set(SECTION_CLASSES):
                found.append(child)
            elif child.tag in {"div", "article", "main"} and not _classes(child) & SKIP_CLASSES:
                found.extend(self._child_sections(child))
        return found

    def _walk_section(
        self, sec: etree._Element, parent_path: str, out: list[SectionContent]
    ) -> None:
        number, title = self._heading(sec)
        label = f"{number} {title}".strip() if number else title
        path = f"{parent_path} > {label}" if parent_path else label
        level = self._level(sec)

        text = "\n\n".join(self._blocks(sec, path))
        if text:
            out.append(
                SectionContent(
                    title=title or label or "Untitled",
                    text=text,
                    level=level,
                    path=path,
                    anchor=sec.get("id"),
                )
            )
        for sub in self._child_sections(sec):
            self._walk_section(sub, path, out)

    @staticmethod
    def _level(sec: etree._Element) -> int:
        cls = _classes(sec)
        for level, name in enumerate(SECTION_CLASSES[:4], start=1):
            if name in cls:
                return level
        return 1

    def _heading(self, sec: etree._Element) -> tuple[str, str]:
        titles = sec.xpath('./*[contains(@class,"ltx_title")][1]')
        if not titles:
            return "", ""
        h = titles[0]
        tag = h.xpath('.//*[contains(@class,"ltx_tag")][1]')
        number = _clean(tag[0].text_content()) if tag else ""
        full = self._inline(h)
        title = full[len(number) :].strip() if number and full.startswith(number) else full
        return number, title.rstrip(".:")

    # ------------------------------------------------------------------ blocks

    def _blocks(self, el: etree._Element, section_path: str) -> list[str]:
        """Render the block-level children of a section/abstract, skipping nested sections."""
        out: list[str] = []
        for child in el:
            if not isinstance(child.tag, str) or child.tag in SKIP_TAGS:
                continue
            cls = _classes(child)
            if child.tag == "section" and cls & set(SECTION_CLASSES):
                continue
            if cls & SKIP_CLASSES or "ltx_title" in " ".join(cls):
                continue
            out.extend(self._render_block(child, section_path))
        return [b for b in out if b]

    def _render_block(self, el: etree._Element, section_path: str) -> list[str]:
        cls = _classes(el)
        if cls & SKIP_CLASSES or el.tag in SKIP_TAGS:
            return []
        if "ltx_equation" in cls or "ltx_equationgroup" in cls or "ltx_eqn_table" in cls:
            return self._equation(el, section_path)
        if el.tag == "figure":
            return self._figure(el, section_path)
        if el.tag == "table" or "ltx_tabular" in cls:
            return [self._table(el)]
        if el.tag in {"ul", "ol"}:
            items = [f"- {self._inline(li)}" for li in el.xpath("./li")]
            return ["\n".join(i for i in items if len(i) > 2)]
        if "ltx_theorem" in " ".join(cls) or any(c.startswith("ltx_theorem") for c in cls):
            return self._theorem(el, section_path)
        if "ltx_listing" in cls or el.tag == "pre":
            return [_clean(el.text_content())]
        if el.tag in {"p"} or not any(
            isinstance(c.tag, str) and c.tag in BLOCK_TAGS for c in el.iterchildren()
        ):
            text = self._inline(el)
            return [text] if text else []
        # a container with block children: an ltx_para holding <p> and equation tables, etc.
        out: list[str] = []
        pending: list[str] = []
        if el.text and el.text.strip():
            pending.append(_esc(el.text))
        for child in el:
            if isinstance(child.tag, str) and child.tag in BLOCK_TAGS:
                if pending:
                    out.append(_clean(" ".join(pending)))
                    pending = []
                out.extend(self._render_block(child, section_path))
            elif isinstance(child.tag, str) and child.tag not in SKIP_TAGS:
                pending.append(self._inline(child, include_tail=False))
            if child.tail and child.tail.strip():
                pending.append(_esc(child.tail))
        if pending:
            out.append(_clean(" ".join(pending)))
        return [b for b in out if b]

    def _theorem(self, el: etree._Element, section_path: str) -> list[str]:
        title = el.xpath('.//*[contains(@class,"ltx_title_theorem")][1]')
        label = _clean(title[0].text_content()) if title else ""
        body: list[str] = []
        for child in el:
            if not isinstance(child.tag, str) or child.tag in SKIP_TAGS:
                continue
            if "ltx_title" in " ".join(_classes(child)):
                continue
            body.extend(self._render_block(child, section_path))
        text = " ".join(b for b in body if b)
        return [f"**{label}** {text}".strip()] if (label or text) else []

    def _figure(self, el: etree._Element, section_path: str) -> list[str]:
        out: list[str] = []
        caption = el.xpath('.//*[self::figcaption or contains(@class,"ltx_caption")][1]')
        cap = self._inline(caption[0]) if caption else ""
        # LaTeXML renders tables either as <table> or, when wrapped by \resizebox/\rotatebox, as
        # nested <span class="ltx_tabular">. Take the outermost of either kind.
        for tbl in el.xpath(
            './/*[contains(concat(" ", @class, " "), " ltx_tabular ")]'
            '[not(ancestor::*[contains(concat(" ", @class, " "), " ltx_tabular ")])]'
        ):
            out.append(self._table(tbl))
        if cap:
            out.insert(0, cap)
        return [b for b in out if b]

    def _table(self, tbl: etree._Element) -> str:
        row_xp = './/tr | .//*[contains(concat(" ", @class, " "), " ltx_tr ")]'
        cell_xp = (
            "./td | ./th | ./*[contains(concat(' ', @class, ' '), ' ltx_td ') "
            "or contains(concat(' ', @class, ' '), ' ltx_th ')]"
        )
        rows: list[str] = []
        for tr in tbl.xpath(row_xp)[:MAX_TABLE_ROWS]:
            cells = [self._inline(c) for c in tr.xpath(cell_xp)]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _equation(self, el: etree._Element, section_path: str) -> list[str]:
        """One record per typeset row: an align block with 3 numbered lines yields 3 equations."""
        rows = el.xpath('.//tr[contains(@class,"ltx_eqn_row")]') or [el]
        blocks: list[str] = []
        for row in rows:
            maths = row.xpath(".//math")
            latex = " ".join(_math_text(m) for m in maths).strip()
            if not latex:
                continue
            tag = row.xpath('.//*[contains(@class,"ltx_tag_equation")][1]')
            number = _clean(tag[0].text_content()) if tag else None
            ident = row.get("id") or el.get("id")
            self._equations.append(
                Equation(
                    ordinal=len(self._equations),
                    latex=latex,
                    number=number,
                    display=True,
                    section_path=section_path,
                    anchor=ident,
                )
            )
            tagged = f"{latex} \\tag{{{number.strip('()')}}}" if number else latex
            blocks.append(f"$${tagged}$$")
        return blocks

    # ------------------------------------------------------------------ inline

    def _inline(self, el: etree._Element, include_tail: bool = False) -> str:
        parts: list[str] = []
        self._inline_into(el, parts, top=True)
        text = _clean("".join(parts))
        if include_tail and el.tail:
            text = _clean(text + " " + _esc(el.tail))
        return _tidy(text)

    def _inline_into(self, el: etree._Element, parts: list[str], top: bool = False) -> None:
        if not isinstance(el.tag, str):
            return
        if not top:
            cls = _classes(el)
            if el.tag in SKIP_TAGS or cls & SKIP_CLASSES:
                return
            if el.tag == "math":
                latex = _math_text(el)
                if latex:
                    delim = "$$" if el.get("display") == "block" else "$"
                    parts.append(f" {delim}{latex}{delim} ")
                return
            if "ltx_note" in cls or "ltx_role_footnote" in cls:
                note = self._inline(el)  # inline rendering keeps math inside footnotes
                if note:
                    parts.append(f" (note: {note}) ")
                return
            if el.tag == "br":
                parts.append(" ")
                return
        if el.text:
            parts.append(_esc(el.text))
        for child in el:
            self._inline_into(child, parts)
            if child.tail:
                parts.append(_esc(child.tail))

    # -------------------------------------------------------------- references

    def _references(self, doc: etree._Element) -> list[Reference]:
        refs: list[Reference] = []
        for i, li in enumerate(doc.xpath('.//li[contains(@class,"ltx_bibitem")]')):
            text = self._inline(li)  # inline rendering keeps math in titles as $...$
            urls = [
                a.get("href")
                for a in li.xpath(".//a[@href]")
                if (a.get("href") or "").startswith("http")
            ]
            refs.append(Reference(ordinal=i, key=li.get("id"), text=text, urls=urls))
        return refs

    # ------------------------------------------------------------ citation contexts

    def _citation_contexts(self, doc: etree._Element) -> list[CitationContext]:
        """The sentence around every in-text citation, with the bibliography keys it cites.

        LaTeXML renders a LaTeX cite as <cite class="ltx_cite"> containing links to "#bib.bibN",
        the id of the bibliography <li>. That link is what lets us say *which sentence of paper B
        talks about paper A* - the raw material for critique detection (ADR 0009).
        """
        contexts: list[CitationContext] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for para in doc.xpath(
            './/p[contains(concat(" ", normalize-space(@class), " "), " ltx_p ")]'
        ):
            if para.xpath('ancestor::*[contains(@class,"ltx_bibliography")]'):
                continue
            cites = para.xpath('.//cite[contains(@class,"ltx_cite")]')
            if not cites:
                continue
            sentences = _SENTENCE_SPLIT.split(self._inline(para))
            anchor = next(iter(para.xpath("ancestor::section[@id][1]/@id")), None)
            for cite in cites:
                keys = tuple(
                    href.split("#", 1)[1] for href in cite.xpath(".//a/@href") if "#bib" in href
                )
                if not keys:
                    continue
                marker = self._inline(cite)
                sentence = next((s for s in sentences if marker and marker in s), None)
                if sentence is None:
                    continue
                sentence = sentence.strip()[:MAX_CONTEXT_CHARS]
                if (sentence, keys) in seen:
                    continue
                seen.add((sentence, keys))
                contexts.append(CitationContext(keys=list(keys), sentence=sentence, anchor=anchor))
        return contexts

    # ------------------------------------------------------------------- stats

    def _stats(self, sections: list[SectionContent], refs: list[Reference]) -> dict:
        text = "\n".join(s.text for s in sections)
        return {
            "words": len(text.split()),
            "sections": sum(1 for s in sections if s.title != "Abstract" and s.level <= 2),
            "display_equations": len(self._equations),
            "inline_math": len(re.findall(r"(?<!\$)\$(?!\$)", text)) // 2,
            "references": len(refs),
        }
