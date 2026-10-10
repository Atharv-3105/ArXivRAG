"""One-off build of the curated foundation corpus (corpus/foundational.csv).

    python -m src.pipeline.build_corpus verify            # check every id against arXiv, no DB
    python -m src.pipeline.build_corpus ingest            # verify, then ingest + embed (resumable)
    python -m src.pipeline.build_corpus ingest --topic agents --limit 20

Why a curated list instead of a search query: the tutor answers questions about the papers the
field is built on (Transformer, BERT, LoRA, DPO, ReAct...), so the corpus is chosen, not sampled.
It also makes retrieval evaluation reproducible: the same ids give the same corpus every time.

Ids are *verified* against the arXiv title before anything is ingested. A wrong id (a typo, a
withdrawn paper, or an id remembered incorrectly) is reported and skipped rather than silently
indexing a different paper under a famous name.

Ingestion is the normal pipeline (`run_pipeline`), so it inherits idempotency: an already-indexed
paper is skipped, a half-written one is repaired, and a parse failure goes to the DLQ without
stopping the run. Re-running after a failure continues where it left off.
"""

import argparse
import asyncio
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.core.config import settings
from src.pipeline.fetcher import ArxivFetcher
from src.pipeline.models import PaperMetadata

logger = structlog.get_logger(__name__)

DEFAULT_LIST = Path("corpus/foundational.csv")
VERIFY_BATCH = 25  # ids per metadata request (arXiv accepts far more; this keeps URLs short)
MATCH_THRESHOLD = 0.8  # share of the expected title tokens that must appear in the real title

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_VERSION = re.compile(r"v\d+$")


@dataclass(frozen=True)
class CorpusEntry:
    arxiv_id: str
    topic: str
    title_match: str


@dataclass(frozen=True)
class Verdict:
    entry: CorpusEntry
    status: str  # "ok" | "title_mismatch" | "not_found"
    actual_title: str | None = None


def load_corpus(path: Path = DEFAULT_LIST) -> list[CorpusEntry]:
    """Read the curated list. Duplicate ids keep their first occurrence."""
    seen: set[str] = set()
    entries: list[CorpusEntry] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            arxiv_id = row["arxiv_id"].strip()
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            entries.append(CorpusEntry(arxiv_id, row["topic"].strip(), row["title_match"].strip()))
    return entries


def _tokens(text: str) -> list[str]:
    return [t for t in _NON_ALNUM.split(text.lower()) if t]


def title_matches(expected: str, actual: str, threshold: float = MATCH_THRESHOLD) -> bool:
    """True when (almost) every token of the expected fragment appears in the real title.

    Token containment rather than string equality: arXiv titles carry subtitles, capitalisation
    and punctuation that differ from how a paper is cited ("BERT: Pre-training of ...")."""
    wanted = _tokens(expected)
    if not wanted:
        return False
    have = set(_tokens(actual))
    return sum(t in have for t in wanted) / len(wanted) >= threshold


def strip_version(arxiv_id: str) -> str:
    return _VERSION.sub("", arxiv_id)


async def verify(fetcher: ArxivFetcher, entries: list[CorpusEntry]) -> list[Verdict]:
    """Fetch metadata for every id and compare titles. Order of `entries` is preserved."""
    titles: dict[str, str] = {}
    for i in range(0, len(entries), VERIFY_BATCH):
        batch = entries[i : i + VERIFY_BATCH]
        papers: list[PaperMetadata] = await fetcher.fetch_papers(
            query=None, id_list=[e.arxiv_id for e in batch]
        )
        for p in papers:
            titles[strip_version(p.arxiv_id)] = p.title

    verdicts: list[Verdict] = []
    for e in entries:
        actual = titles.get(e.arxiv_id)
        if actual is None:
            verdicts.append(Verdict(e, "not_found"))
        elif title_matches(e.title_match, actual):
            verdicts.append(Verdict(e, "ok", actual))
        else:
            verdicts.append(Verdict(e, "title_mismatch", actual))
    return verdicts


def select(
    entries: list[CorpusEntry], topic: str | None = None, limit: int | None = None
) -> list[CorpusEntry]:
    chosen = [e for e in entries if topic is None or e.topic == topic]
    return chosen[:limit] if limit else chosen


def print_verification(verdicts: list[Verdict]) -> None:
    ok = [v for v in verdicts if v.status == "ok"]
    bad = [v for v in verdicts if v.status != "ok"]
    topics: dict[str, int] = {}
    for v in ok:
        topics[v.entry.topic] = topics.get(v.entry.topic, 0) + 1
    print(f"\nverified {len(ok)}/{len(verdicts)} ids")
    print("  " + ", ".join(f"{t} {n}" for t, n in sorted(topics.items())))
    for v in bad:
        print(
            f"  SKIP {v.entry.arxiv_id} [{v.status}] wanted '{v.entry.title_match}'"
            + (f" but arXiv has '{v.actual_title}'" if v.actual_title else "")
        )


async def ingest(entries: list[CorpusEntry], batch_size: int, reingest: bool) -> None:
    from src.pipeline.runner import run_pipeline  # imports the DB/vector stack: only when needed

    totals = {
        "fetched": 0,
        "parsed": 0,
        "html": 0,
        "pdf": 0,
        "indexed": 0,
        "skipped": 0,
        "repaired": 0,
        "failed": 0,
    }  # "indexed" counts chunks written
    for i in range(0, len(entries), batch_size):
        ids = [e.arxiv_id for e in entries[i : i + batch_size]]
        metrics = await run_pipeline(ids=ids, max_papers=len(ids), reingest=reingest)
        for k in totals:
            totals[k] += getattr(metrics, k, 0)
        print(f"batch {i // batch_size + 1}: {len(ids)} ids -> {totals}", flush=True)
    print(f"\ndone: {totals}")
    print("failed papers are in the failed_papers table; re-run the same command to retry them")


async def _main(args: argparse.Namespace) -> int:
    entries = select(load_corpus(Path(args.list)), args.topic, args.limit)
    if not entries:
        print("no entries selected")
        return 1
    fetcher = ArxivFetcher()
    try:
        verdicts = await verify(fetcher, entries)
    finally:
        await fetcher.aclose()
    print_verification(verdicts)

    if args.command == "verify":
        return 0 if all(v.status == "ok" for v in verdicts) else 2
    good = [v.entry for v in verdicts if v.status == "ok"]
    print(
        f"\ningesting {len(good)} papers with {settings.EMBEDDING_MODEL} "
        f"into collection {settings.QDRANT_COLLECTION}"
    )
    await ingest(good, args.batch, args.reingest)
    return 0


def main() -> None:
    cli = argparse.ArgumentParser(description="Build the curated foundation corpus")
    cli.add_argument("command", choices=["verify", "ingest"])
    cli.add_argument("--list", default=str(DEFAULT_LIST), help="CSV of arxiv_id,topic,title_match")
    cli.add_argument("--topic", help="only this topic (e.g. agents, retrieval, alignment)")
    cli.add_argument("--limit", type=int, help="only the first N selected entries")
    cli.add_argument("--batch", type=int, default=10, help="papers per pipeline run")
    cli.add_argument("--reingest", action="store_true", help="re-parse and re-embed indexed papers")
    raise SystemExit(asyncio.run(_main(cli.parse_args())))


if __name__ == "__main__":
    main()
