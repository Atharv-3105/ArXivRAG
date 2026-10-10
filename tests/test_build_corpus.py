"""The curated corpus list and its verification (no network: arXiv metadata is faked)."""

import re
from datetime import UTC, datetime
from pathlib import Path

from src.pipeline.build_corpus import (
    DEFAULT_LIST,
    CorpusEntry,
    load_corpus,
    select,
    strip_version,
    title_matches,
    verify,
)
from src.pipeline.models import PaperMetadata

_ID_SHAPE = re.compile(r"^\d{4}\.\d{4,5}$")


def _paper(arxiv_id: str, title: str) -> PaperMetadata:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return PaperMetadata(
        arxiv_id=arxiv_id, title=title, authors=["A"], abstract="x", published=now,
        updated=now, categories=["cs.LG"], pdf_url="http://x/pdf", primary_category="cs.LG",
    )  # fmt: skip


class FakeFetcher:
    def __init__(self, papers: list[PaperMetadata]):
        self.papers = papers
        self.requests: list[list[str]] = []

    async def fetch_papers(self, query, id_list=None, **_):
        self.requests.append(id_list)
        return [p for p in self.papers if strip_version(p.arxiv_id) in id_list]


def test_title_match_tolerates_subtitles_case_and_punctuation():
    real = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    assert title_matches("bert pre-training of deep bidirectional", real)
    assert title_matches("attention is all you need", "Attention Is All You Need")


def test_title_match_rejects_a_different_paper():
    assert not title_matches("attention is all you need", "Graph Attention Networks")
    assert not title_matches("", "Anything")


def test_strip_version():
    assert strip_version("1706.03762v7") == "1706.03762"
    assert strip_version("1706.03762") == "1706.03762"


async def test_verify_flags_wrong_and_missing_ids():
    entries = [
        CorpusEntry("1706.03762", "transformers", "attention is all you need"),
        CorpusEntry("1810.04805", "nlp-pretraining", "bert pre-training"),  # arXiv says otherwise
        CorpusEntry("9999.99999", "agents", "a paper that does not exist"),
    ]
    fetcher = FakeFetcher(
        [
            _paper("1706.03762v7", "Attention Is All You Need"),
            _paper("1810.04805v2", "A completely different paper"),
        ]
    )
    verdicts = await verify(fetcher, entries)  # type: ignore[arg-type]
    assert [v.status for v in verdicts] == ["ok", "title_mismatch", "not_found"]
    assert verdicts[1].actual_title == "A completely different paper"


async def test_verify_batches_requests():
    entries = [CorpusEntry(f"2401.{i:05d}", "t", "x") for i in range(60)]
    fetcher = FakeFetcher([])
    await verify(fetcher, entries)  # type: ignore[arg-type]
    assert [len(r) for r in fetcher.requests] == [25, 25, 10]


def test_select_filters_by_topic_and_limit():
    entries = [CorpusEntry(f"2401.{i:05d}", "a" if i % 2 else "b", "x") for i in range(10)]
    assert {e.topic for e in select(entries, topic="a")} == {"a"}
    assert len(select(entries, limit=3)) == 3


def test_curated_list_is_well_formed():
    """Every id is well formed, unique, titled and topic-tagged: a typo here would otherwise
    only surface during a 40-minute ingest."""
    path = Path(__file__).resolve().parent.parent / DEFAULT_LIST
    entries = load_corpus(path)
    ids = [e.arxiv_id for e in entries]
    assert len(ids) == len(set(ids))
    assert 100 <= len(entries)
    for e in entries:
        assert e.title_match and e.topic
        assert len(_ID_SHAPE.findall(e.arxiv_id)) == 1, e.arxiv_id
    assert {"agents", "retrieval", "alignment"} <= {e.topic for e in entries}
