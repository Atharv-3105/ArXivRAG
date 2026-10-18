import re
import uuid
from collections.abc import Callable

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
    """Sliding-window chunker.

    Size is measured in EMBEDDING-MODEL TOKENS when a `token_counter` is given, and in characters
    otherwise. Tokens are the honest unit: the model ignores everything past its window without an
    error, and 800 characters of LaTeX-heavy text can be 400 tokens against a 256-token window
    (measured: 32% of chunks overflowed before this). The counter works word by word, which is exact
    for WordPiece (it splits on whitespace first), so a running sum equals the chunk's token count.

    Formulas are never split. A single formula longer than the chunk size becomes an oversized chunk
    rather than a broken one, because half a formula is worse than a long chunk.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        token_counter: Callable[[str], int] | None = None,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ):
        self._count = token_counter
        if token_counter is not None:
            self.chunk_size = max_tokens or settings.CHUNK_MAX_TOKENS
            self.chunk_overlap = (
                settings.CHUNK_OVERLAP_TOKENS if overlap_tokens is None else overlap_tokens
            )
        else:
            self.chunk_size = chunk_size or settings.CHUNK_SIZE
            self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def _cost(self, word: str) -> int:
        return self._count(word) if self._count else len(word) + 1

    def chunk_sections(self, paper_id: str, sections: list[SectionContent]) -> list[Chunk]:
        all_chunks: list[Chunk] = []

        for section_index, section in enumerate(sections):
            if len(section.text) < MIN_SECTION_CHARS:
                continue

            title = (section.title or "Unknown")[:255]  # paragraph run-in headings can be long
            context = section.path or title
            # The section path is embedded with every chunk (Chunk.embedding_text), so it spends
            # part of the model's window too.
            reserved = sum(self._cost(w) for w in context.split()) + 2 if self._count else 0
            for i, text in enumerate(self._split_text(section.text, reserved)):
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

    def _split_text(self, text: str, reserved: int = 0) -> list[str]:
        # Never reserve so much that a chunk cannot hold more than its own overlap.
        limit = max(self.chunk_size - reserved, self.chunk_overlap + 16)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        fresh_words = 0  # tokens added since the last emitted chunk

        def emit() -> tuple[list[str], int]:
            """Close the current chunk; carry its tail (up to the overlap) into the next one."""
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_len = 0
            for w in reversed(current):
                cost = self._cost(w)
                if tail_len + cost > self.chunk_overlap:
                    break
                tail.insert(0, w)
                tail_len += cost
            return tail, tail_len

        for token in _TOKEN.findall(text):
            cost = self._cost(token)
            if self._count and fresh_words and current_len + cost > limit:
                # Token budget: close BEFORE the word that would overflow the model's window.
                current, current_len = emit()
                fresh_words = 0
            current.append(token)
            current_len += cost
            fresh_words += 1

            if not self._count and current_len >= self.chunk_size:
                current, current_len = emit()  # character mode: close once the size is reached
                fresh_words = 0

        if fresh_words:  # remainder holds tokens that no emitted chunk contains yet
            chunks.append(" ".join(current))

        return chunks
