import re
import uuid

from src.core.config import settings
from src.pipeline.models import Chunk, SectionContent

# Fixed namespace: the same (paper, section, position) always maps to the same chunk id, so
# re-ingesting a paper overwrites its points/rows instead of duplicating them.
_CHUNK_NAMESPACE = uuid.UUID("6f1d2c3e-5a7b-4c8d-9e0f-a1b2c3d4e5f6")

MIN_SECTION_CHARS = 100

# An escaped dollar (`\$`, a literal price in prose) neither opens nor closes a span.
_MATH = r"(?<!\\)\$\$.+?(?<!\\)\$\$|(?<!\\)\$.+?(?<!\\)\$"
# A token is a run of characters in which a whole `$...$` / `$$...$$` span counts as ONE unit, so a
# chunk boundary can never fall inside a formula (and math containing spaces stays in one piece).
_TOKEN = re.compile(rf"(?:{_MATH}|\S)+", re.DOTALL)
_MATH_SPAN = re.compile(_MATH, re.DOTALL)


class TextChunker:
    """Sliding-window chunker. Size and overlap are both measured in CHARACTERS.

    Formulas are never split. A single formula longer than the chunk size becomes an oversized chunk
    rather than a broken one, because half a formula is worse than a long chunk.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk_sections(self, paper_id: str, sections: list[SectionContent]) -> list[Chunk]:
        all_chunks: list[Chunk] = []

        for section_index, section in enumerate(sections):
            if len(section.text) < MIN_SECTION_CHARS:
                continue

            title = (section.title or "Unknown")[:255]  # paragraph run-in headings can be long
            context = section.path or title
            for i, text in enumerate(self._split_text(section.text)):
                chunk_id = uuid.uuid5(_CHUNK_NAMESPACE, f"{paper_id}:{section_index}:{i}")
                all_chunks.append(
                    Chunk(
                        chunk_id=str(chunk_id),
                        paper_id=paper_id,
                        section_title=title,
                        content=text,
                        context=context,
                        metadata={
                            "word_count": len(text.split()),
                            "section_index": section_index,
                            "chunk_index": i,
                            "section_path": context,
                            "anchor": section.anchor,
                            "page_start": section.page_start,
                            "page_end": section.page_end,
                            "math_spans": len(_MATH_SPAN.findall(text)),
                        },
                    )
                )

        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        fresh_words = 0  # tokens added since the last emitted chunk

        for token in _TOKEN.findall(text):
            current.append(token)
            current_len += len(token) + 1
            fresh_words += 1

            if current_len >= self.chunk_size:
                chunks.append(" ".join(current))
                # Carry over the tail (up to chunk_overlap chars) as context for the next chunk.
                tail: list[str] = []
                tail_len = 0
                for w in reversed(current):
                    if tail_len + len(w) + 1 > self.chunk_overlap:
                        break
                    tail.insert(0, w)
                    tail_len += len(w) + 1
                current, current_len, fresh_words = tail, tail_len, 0

        if fresh_words:  # remainder holds tokens that no emitted chunk contains yet
            chunks.append(" ".join(current))

        return chunks
