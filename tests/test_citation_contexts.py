"""Citation contexts: parsing the sentence around each in-text cite, classifying what it does,
and resolving it to arXiv papers."""

import pytest

from src.pipeline.citations import citation_function, context_rows
from src.pipeline.html_parser import HtmlParser
from src.pipeline.models import CitationContext, Reference
from tests.test_html_parser import FILLER, page, para, section


def cite(*keys: str, text: str = "[1]") -> str:
    links = "".join(f'<a href="#bib.{k}" class="ltx_ref">{text}</a>' for k in keys)
    return f'<cite class="ltx_cite ltx_citemacro_cite">{links}</cite>'


def bibitem(key: str, body: str) -> str:
    return f'<li id="bib.{key}" class="ltx_bibitem"><span class="ltx_bibblock">{body}</span></li>'


def _parse(*body: str, refs: str = ""):
    return HtmlParser(min_words=1).parse(page(*body, refs=refs))


def test_the_sentence_around_each_cite_is_captured_with_its_keys_and_section():
    doc = _parse(
        section(
            "S2",
            "2",
            "Related Work",
            para(
                f"Transformers are popular. LoRA {cite('bib1', text='[1]')} fails to adapt "
                f"embeddings. We follow QLoRA {cite('bib2', text='[2]')} for quantization. {FILLER}"
            ),
        ),
        refs=bibitem("bib1", "Hu et al. LoRA. arXiv:2106.09685") + bibitem("bib2", "QLoRA"),
    )
    [first, second] = doc.citation_contexts
    assert first.keys == ["bib.bib1"] and first.sentence == "LoRA [1] fails to adapt embeddings."
    assert first.anchor == "S2"
    assert second.keys == ["bib.bib2"] and second.sentence.startswith("We follow QLoRA [2]")
    assert doc.stats["citation_contexts"] == 2


def test_one_cite_with_several_keys():
    doc = _parse(
        section(
            "S1", "1", "Intro", para(f"Prior work {cite('bib1', 'bib2')} studied this. {FILLER}")
        )
    )
    assert doc.citation_contexts[0].keys == ["bib.bib1", "bib.bib2"]


def test_cites_inside_the_bibliography_are_ignored():
    refs = bibitem("bib1", f'<p class="ltx_p">See also {cite("bib2")}.</p>')
    doc = _parse(section("S1", "1", "Intro", para(FILLER)), refs=refs)
    assert doc.citation_contexts == []


@pytest.mark.parametrize(
    ("sentence", "function"),
    [
        ("LoRA [3] fails to capture full-rank updates.", "critical"),
        ("As shown by [4], the method suffers from instability.", "critical"),
        ("Their analysis is limited to linear models [2].", "critical"),
        ("Our model outperforms BERT [5] on GLUE.", "comparative"),
        ("Unlike [6], we train end to end.", "comparative"),
        ("We follow the setup of [7].", "builds_on"),
        ("Our encoder is based on the Transformer [1].", "builds_on"),
        ("Attention was introduced in [1].", "builds_on"),
        ("Language models have grown quickly [8, 9].", "neutral"),
        ("However, we use a larger batch [2].", "builds_on"),  # "however" alone is not criticism
    ],
)
def test_citation_function(sentence, function):
    assert citation_function(sentence) == function


def test_context_rows_resolve_keys_to_arxiv_papers():
    refs = [
        Reference(ordinal=0, key="bib.bib1", text="Hu et al. LoRA. arXiv:2106.09685"),
        Reference(ordinal=1, key="bib.bib2", text="A journal paper without an arXiv id."),
        Reference(ordinal=2, key="bib.bib3", text="Our own earlier version, arXiv:2401.00001v1"),
    ]
    contexts = [
        CitationContext(
            keys=["bib.bib1", "bib.bib2"], sentence="LoRA [1] fails badly.", anchor="S2"
        ),
        CitationContext(keys=["bib.bib3"], sentence="Extending our prior work [3].", anchor="S1"),
        CitationContext(keys=["bib.nope"], sentence="Unknown key [9].", anchor=None),
    ]
    [row] = context_rows("2401.00001v2", refs, contexts)
    assert row == {
        "citing_id": "2401.00001v2",
        "cited_base_id": "2106.09685",
        "ref_ordinal": 0,
        "sentence": "LoRA [1] fails badly.",
        "anchor": "S2",
        "function": "critical",
    }
