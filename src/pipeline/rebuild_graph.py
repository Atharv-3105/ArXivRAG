"""Rebuild the citation graph for papers that are already stored.

    python -m src.pipeline.rebuild_graph            # edges + contexts + influence
    python -m src.pipeline.rebuild_graph --edges-only

Edges and citation contexts are normally written when a paper is ingested. A paper stored before
the graph existed (or by an older pipeline) has references but no edges, and `reindex` only
re-embeds, so its place in the graph stayed empty. Both are derived data, so both are rebuilt here
without touching arXiv:

* edges: from the stored `paper_references` (pure database work);
* contexts: by re-parsing the cached arXiv HTML (the PDF tier has no link from "[12]" to a
  bibliography entry, so PDF papers contribute edges only, and only if they stored references).

Idempotent: edges use ON CONFLICT DO NOTHING and a paper's contexts are replaced as a unit.
"""

import argparse
import asyncio
from collections import defaultdict
from pathlib import Path

import structlog
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from src.core.config import settings
from src.core.logging import setup_logging
from src.pipeline.citations import citation_edges, context_rows
from src.pipeline.extractor import _safe_name
from src.pipeline.html_parser import HtmlParser
from src.pipeline.models import Reference
from src.storage.database import (
    CitationContextRow,
    Paper,
    PaperCitation,
    PaperReference,
    dispose_engine,
    get_async_session,
    init_db_schema,
)

logger = structlog.get_logger(__name__)


def edges_from_stored(rows) -> list[dict]:
    """`rows`: (paper_id, ordinal, ref_key, text, urls) tuples, any order, any number of papers."""
    by_paper: dict[str, list[Reference]] = defaultdict(list)
    for paper_id, ordinal, key, text, urls in rows:
        by_paper[paper_id].append(Reference(ordinal=ordinal, key=key, text=text, urls=urls or []))
    edges: list[dict] = []
    for paper_id, refs in by_paper.items():
        edges.extend(citation_edges(paper_id, sorted(refs, key=lambda r: r.ordinal)))
    return edges


async def rebuild(cache_dir: str | None = None, edges_only: bool = False) -> dict:
    await init_db_schema()
    html_dir = Path(cache_dir or settings.CACHE_DIR) / "html"
    report = {"papers": 0, "edges": 0, "context_rows": 0, "html_missing": 0, "html_unparsable": 0}
    parser = HtmlParser()
    try:
        async with get_async_session() as session:
            papers = (await session.execute(select(Paper.arxiv_id, Paper.parse_method))).all()
            refs = (
                await session.execute(
                    select(
                        PaperReference.paper_id,
                        PaperReference.ordinal,
                        PaperReference.ref_key,
                        PaperReference.text,
                        PaperReference.urls,
                    )
                )
            ).all()
        report["papers"] = len(papers)

        edges = edges_from_stored(refs)
        async with get_async_session() as session:
            for i in range(0, len(edges), 500):
                await session.execute(
                    insert(PaperCitation).values(edges[i : i + 500]).on_conflict_do_nothing()
                )
            await session.commit()
        report["edges"] = len(edges)

        if not edges_only:
            for arxiv_id, method in papers:
                if method != "html":
                    continue
                path = html_dir / f"{_safe_name(arxiv_id)}.html"
                if not path.exists():
                    report["html_missing"] += 1
                    continue
                try:
                    doc = await asyncio.to_thread(parser.parse, path.read_text("utf-8"))
                except Exception as e:  # noqa: BLE001 - one bad file must not stop the rest
                    logger.warning(
                        "Could not re-parse cached HTML", arxiv_id=arxiv_id, error=str(e)
                    )
                    report["html_unparsable"] += 1
                    continue
                rows = context_rows(arxiv_id, doc.references, doc.citation_contexts)
                async with get_async_session() as session:
                    await session.execute(
                        delete(CitationContextRow).where(CitationContextRow.citing_id == arxiv_id)
                    )
                    if rows:
                        await session.execute(insert(CitationContextRow).values(rows))
                    await session.commit()
                report["context_rows"] += len(rows)

        from src.graph.influence import refresh_influence
        from src.graph.store import PgGraphStore

        await refresh_influence(PgGraphStore())
    finally:
        await dispose_engine()
    logger.info("Citation graph rebuilt", **report)
    return report


def main() -> None:
    setup_logging()
    cli = argparse.ArgumentParser(
        description="Rebuild citation edges and contexts from stored data"
    )
    cli.add_argument("--edges-only", action="store_true", help="skip re-parsing cached HTML")
    cli.add_argument("--cache-dir", help="where HTML is cached (default: CACHE_DIR)")
    args = cli.parse_args()
    print(asyncio.run(rebuild(args.cache_dir, args.edges_only)))


if __name__ == "__main__":
    main()
