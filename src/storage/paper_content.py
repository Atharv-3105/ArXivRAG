"""Read a paper in order: sections, their chunks, and numbered equations (for the Tutor's guided
reading and equation walk-through modes).

Vector search answers "what is relevant"; reading needs "what comes next", so this reads Postgres
directly. Reading order is the parser's: chunks carry `section_index` and `chunk_index` in their
metadata (see TextChunker).
"""

from typing import Protocol

from sqlalchemy import text

from src.storage.database import get_async_session

_META = text(
    """
    SELECT arxiv_id, title, parse_method, math_fidelity, published_date
    FROM papers WHERE arxiv_id = :p OR base_id = :p
    ORDER BY arxiv_id DESC LIMIT 1
    """
)
_SECTIONS = text(
    """
    SELECT DISTINCT ON (idx) idx, title, path, anchor FROM (
        SELECT CAST(metadata->>'section_index' AS INT) AS idx, section_title AS title,
               metadata->>'section_path' AS path, metadata->>'anchor' AS anchor
        FROM paper_chunks WHERE paper_id = :p
    ) s
    ORDER BY idx
    """
)
_SECTION_CHUNKS = text(
    """
    SELECT chunk_id, content, section_title, metadata->>'anchor' AS anchor
    FROM paper_chunks
    WHERE paper_id = :p AND CAST(metadata->>'section_index' AS INT) = :i
    ORDER BY CAST(metadata->>'chunk_index' AS INT)
    LIMIT :limit
    """
)
# "equation 3" means the typeset label (3); fall back to position for unnumbered ones.
_EQUATION = text(
    """
    SELECT ordinal, latex, number, section_path, anchor
    FROM paper_equations
    WHERE paper_id = :p AND (number = :label OR number = :bare OR ordinal = :ordinal)
    ORDER BY (number = :label OR number = :bare) DESC, ordinal
    LIMIT 1
    """
)


class PaperContent(Protocol):
    async def paper_meta(self, paper_id: str) -> dict | None: ...

    async def sections(self, paper_id: str) -> list[dict]: ...

    async def section_chunks(self, paper_id: str, index: int, limit: int) -> list[dict]: ...

    async def equation(self, paper_id: str, ref: str) -> dict | None: ...


class PgPaperContent:
    async def paper_meta(self, paper_id):
        async with get_async_session() as s:
            row = (await s.execute(_META, {"p": paper_id})).mappings().one_or_none()
        if row is None:
            return None
        data = dict(row)
        published = data.pop("published_date")
        data["published"] = published.date().isoformat() if published else None
        return data

    async def sections(self, paper_id):
        async with get_async_session() as s:
            return [dict(r) for r in (await s.execute(_SECTIONS, {"p": paper_id})).mappings()]

    async def section_chunks(self, paper_id, index, limit):
        async with get_async_session() as s:
            rows = await s.execute(_SECTION_CHUNKS, {"p": paper_id, "i": index, "limit": limit})
            return [dict(r) for r in rows.mappings()]

    async def equation(self, paper_id, ref):
        ref = ref.strip().strip("()")
        ordinal = int(ref) - 1 if ref.isdigit() else -1
        async with get_async_session() as s:
            row = (
                (
                    await s.execute(
                        _EQUATION,
                        {"p": paper_id, "label": f"({ref})", "bare": ref, "ordinal": ordinal},
                    )
                )
                .mappings()
                .one_or_none()
            )
        return dict(row) if row else None


class InMemoryPaperContent:
    """papers: id -> {"title", "published", "sections": [{"title", "path", "anchor",
    "chunks": [str]}], "equations": [{"latex", "number", "section_path"}]}"""

    def __init__(self) -> None:
        self.papers: dict[str, dict] = {}

    def add(
        self,
        paper_id: str,
        title: str,
        sections: list[dict],
        equations=None,
        published: str = "2024-01-01",
    ) -> None:
        self.papers[paper_id] = {
            "title": title,
            "published": published,
            "sections": sections,
            "equations": equations or [],
        }

    async def paper_meta(self, paper_id):
        p = self.papers.get(paper_id)
        if p is None:
            return None
        return {
            "arxiv_id": paper_id,
            "title": p["title"],
            "parse_method": "html",
            "math_fidelity": "latex",
            "published": p["published"],
        }

    async def sections(self, paper_id):
        p = self.papers.get(paper_id, {"sections": []})
        return [
            {
                "idx": i,
                "title": s["title"],
                "path": s.get("path", s["title"]),
                "anchor": s.get("anchor"),
            }
            for i, s in enumerate(p["sections"])
        ]

    async def section_chunks(self, paper_id, index, limit):
        section = self.papers[paper_id]["sections"][index]
        return [
            {
                "chunk_id": f"{paper_id}:{index}:{i}",
                "content": c,
                "section_title": section["title"],
                "anchor": section.get("anchor"),
            }
            for i, c in enumerate(section["chunks"][:limit])
        ]

    async def equation(self, paper_id, ref):
        ref = ref.strip().strip("()")
        equations = self.papers.get(paper_id, {"equations": []})["equations"]
        for i, eq in enumerate(equations):
            if eq.get("number") in (f"({ref})", ref):
                return {"ordinal": i, "anchor": None, **eq}
        if ref.isdigit() and 0 < int(ref) <= len(equations):
            return {"ordinal": int(ref) - 1, "anchor": None, **equations[int(ref) - 1]}
        return None
