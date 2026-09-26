import pytest

from src.pipeline.citations import base_id, citation_edges, extract_arxiv_ids
from src.pipeline.models import Reference


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A. Vaswani et al. Attention is all you need. arXiv:1706.03762, 2017.", ["1706.03762"]),
        ("arXiv preprint arXiv:2401.12345v2", ["2401.12345"]),  # version dropped
        ("See https://arxiv.org/abs/2305.14314 for details", ["2305.14314"]),
        ("https://arxiv.org/pdf/2305.14314v3", ["2305.14314"]),
        ("broken url arxiv. org/abs/2305.14314", ["2305.14314"]),  # PDF extraction artefact
        ("doi:10.48550/arXiv.2211.03154", ["2211.03154"]),
        ("E. Witten. arXiv:hep-th/9802150", ["hep-th/9802150"]),  # old-style id
        ("ArXiv, vol. abs/2009.04372, 2020", ["2009.04372"]),  # Semantic Scholar style
        ("arXiv:1706.03762 and arXiv:1810.04805", ["1706.03762", "1810.04805"]),
    ],
)
def test_extracts_arxiv_ids(text, expected):
    assert extract_arxiv_ids(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "J. Smith. Some journal, 12(3):2401.12345, 2020.",  # bare number: page range, not an id
        "Proc. NeurIPS, pages 1706.03762-1706.03799",
        "vol. abs/2009.04372",  # 'abs/' only trusted when the reference says arXiv
        "No identifiers here at all.",
    ],
)
def test_ignores_numbers_that_are_not_marked_as_arxiv(text):
    assert extract_arxiv_ids(text) == []


def test_ids_in_reference_urls_count():
    assert extract_arxiv_ids("Some title.", ["https://arxiv.org/abs/2106.09685"]) == ["2106.09685"]


def test_duplicates_are_collapsed_in_order_of_appearance():
    text = "arXiv:1810.04805. https://arxiv.org/abs/1810.04805v2 arXiv:1706.03762"
    assert extract_arxiv_ids(text) == ["1810.04805", "1706.03762"]


def test_base_id():
    assert base_id("2401.12345v12") == "2401.12345"
    assert base_id("hep-th/9802150v1") == "hep-th/9802150"
    assert base_id("2401.12345") == "2401.12345"


def test_citation_edges_dedupe_and_drop_self_citations():
    refs = [
        Reference(ordinal=1, text="Attention. arXiv:1706.03762"),
        Reference(ordinal=2, text="Our earlier version, arXiv:2401.12345v1"),  # self
        Reference(ordinal=3, text="Attention again, arxiv.org/abs/1706.03762v5"),  # duplicate
        Reference(ordinal=4, text="A book with no arXiv id."),
        Reference(ordinal=5, text="BERT", urls=["https://arxiv.org/abs/1810.04805"]),
    ]
    assert citation_edges("2401.12345v2", refs) == [
        {"citing_id": "2401.12345v2", "cited_base_id": "1706.03762", "ref_ordinal": 1},
        {"citing_id": "2401.12345v2", "cited_base_id": "1810.04805", "ref_ordinal": 5},
    ]


def test_no_references_no_edges():
    assert citation_edges("2401.12345v1", []) == []
