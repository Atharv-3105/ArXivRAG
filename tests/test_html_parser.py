"""HtmlParser against fixtures that mimic real LaTeXML output (structure verified on live arXiv pages)."""

import pytest

from src.core.exceptions import ParseError
from src.pipeline.html_parser import HtmlParser

FILLER = " ".join(f"word{i}" for i in range(40))


def math(alt: str, display: str = "inline") -> str:
    # LaTeXML wraps MathML around the glyphs; the LaTeX source lives in `alttext`.
    return (
        f'<math class="ltx_Math" alttext="{alt}" display="{display}"><semantics>'
        f"<mi>GLYPHS-NOT-LATEX</mi></semantics></math>"
    )


def para(*inner: str, pid: str = "S1.p1") -> str:
    return f'<div class="ltx_para" id="{pid}"><p class="ltx_p">{" ".join(inner)}</p></div>'


def equation(alt: str, number: str | None, eid: str = "S1.E1") -> str:
    tag = (
        f'<td class="ltx_eqn_cell ltx_eqn_eqno"><span class="ltx_tag ltx_tag_equation">{number}</span></td>'
        if number
        else ""
    )
    return (
        f'<table class="ltx_equation ltx_eqn_table" id="{eid}"><tbody>'
        f'<tr class="ltx_equation ltx_eqn_row" id="{eid}r">{tag}'
        f'<td class="ltx_eqn_cell">{math(alt, "block")}</td></tr></tbody></table>'
    )


def align(rows: list[tuple[str, str]], eid: str = "S1.E2") -> str:
    trs = "".join(
        f'<tr class="ltx_equation ltx_eqn_row" id="{eid}.{i}">'
        f'<td class="ltx_eqn_cell ltx_eqn_eqno"><span class="ltx_tag ltx_tag_equation">{n}</span></td>'
        f'<td class="ltx_eqn_cell">{math(a, "block")}</td></tr>'
        for i, (a, n) in enumerate(rows)
    )
    return f'<table class="ltx_equationgroup ltx_eqn_table" id="{eid}"><tbody>{trs}</tbody></table>'


def section(sid: str, number: str, title: str, *body: str, kind: str = "ltx_section") -> str:
    h = "h2" if kind == "ltx_section" else "h3"
    return (
        f'<section class="ltx_section_wrapper {kind}" id="{sid}">'
        f'<{h} class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">{number} </span>{title}</{h}>'
        f"{''.join(body)}</section>"
    )


def page(*parts: str, abstract: str = "An abstract about things.", refs: str = "") -> str:
    biblio = (
        f'<section class="ltx_bibliography"><ul class="ltx_biblist">{refs}</ul></section>'
        if refs
        else ""
    )
    return (
        '<html><body><article class="ltx_document">'
        '<h1 class="ltx_title ltx_title_document">A Paper</h1>'
        '<div class="ltx_authors"><span>Ada Lovelace</span></div>'
        f'<div class="ltx_abstract"><h6 class="ltx_title">Abstract</h6><p class="ltx_p">{abstract}</p></div>'
        f"{''.join(parts)}{biblio}</article></body></html>"
    )


@pytest.fixture
def parser():
    return HtmlParser(min_words=5)


def _sections(parser, *parts, **kw):
    return parser.parse(page(*parts, **kw))


# ---------------------------------------------------------------- structure


def test_section_hierarchy_paths_levels_and_anchors(parser):
    doc = _sections(
        parser,
        section(
            "S1",
            "1",
            "Introduction",
            para(FILLER),
            section("S1.SS1", "1.1", "Background", para(FILLER), kind="ltx_subsection"),
        ),
        section("S2", "2", "Method", para(FILLER)),
    )
    by_path = {s.path: s for s in doc.sections}
    assert list(by_path) == [
        "Abstract",
        "1 Introduction",
        "1 Introduction > 1.1 Background",
        "2 Method",
    ]
    assert by_path["1 Introduction > 1.1 Background"].level == 2
    assert by_path["1 Introduction > 1.1 Background"].anchor == "S1.SS1"
    assert by_path["2 Method"].title == "Method"  # the "2" tag is stripped from the title


def test_document_reports_html_method_and_latex_fidelity(parser):
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER)))
    assert doc.method == "html"
    assert doc.math_fidelity == "latex"


# --------------------------------------------------------------------- math


def test_inline_math_is_taken_from_alttext_not_glyphs(parser):
    doc = _sections(
        parser,
        section("S1", "1", "Intro", para("Energy is", math("E=mc^{2}"), "in vacuum.", FILLER)),
    )
    text = doc.sections[1].text
    assert "$E=mc^{2}$" in text
    assert "GLYPHS-NOT-LATEX" not in text


def test_display_equation_is_kept_with_its_number_and_recorded(parser):
    doc = _sections(
        parser,
        section(
            "S1",
            "1",
            "Intro",
            para("We have", FILLER),
            equation(r"\frac{a}{b}=c", "(3)", eid="S1.E3"),
            para("where c is a constant.", pid="S1.p2"),
        ),
    )
    text = doc.sections[1].text
    assert r"$$\frac{a}{b}=c \tag{3}$$" in text
    # reading order is preserved: sentence, equation, continuation
    assert text.index("We have") < text.index(r"\frac{a}{b}") < text.index("where c is a constant")
    [eq] = doc.equations
    assert (eq.latex, eq.number, eq.anchor, eq.section_path) == (
        r"\frac{a}{b}=c",
        "(3)",
        "S1.E3r",
        "1 Intro",
    )


def test_each_row_of_a_multiline_equation_is_its_own_record(parser):
    doc = _sections(
        parser,
        section("S1", "1", "Intro", para(FILLER), align([("a=b", "(1)"), ("c=d", "(2)")])),
    )
    assert [(e.latex, e.number) for e in doc.equations] == [("a=b", "(1)"), ("c=d", "(2)")]
    assert [e.ordinal for e in doc.equations] == [0, 1]


def test_unnumbered_equation_has_no_tag(parser):
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER), equation("x=1", None)))
    assert doc.equations[0].number is None
    assert "$$x=1$$" in doc.sections[1].text


def test_spacing_next_to_math_is_tidied_without_touching_the_formula(parser):
    doc = _sections(
        parser, section("S1", "1", "Intro", para("value (", math(r"a \, b"), ") and", FILLER))
    )
    text = doc.sections[1].text
    assert "($a \\, b$)" in text  # no space inside the parentheses, formula spacing untouched


# ----------------------------------------------------------- other content


def test_theorem_is_rendered_with_its_label(parser):
    thm = (
        '<div class="ltx_theorem ltx_theorem_theorem" id="S1.Thmtheorem1">'
        '<h6 class="ltx_title ltx_title_theorem">Theorem 1.1.</h6>'
        f'<div class="ltx_para"><p class="ltx_p">Let {math("x>0")}. Then {FILLER}</p></div></div>'
    )
    doc = _sections(parser, section("S1", "1", "Intro", thm))
    assert "**Theorem 1.1.** Let $x>0$." in doc.sections[1].text


def test_html_table_becomes_pipe_separated_rows(parser):
    tbl = (
        '<figure class="ltx_table"><figcaption class="ltx_caption">Table 1: Scores.</figcaption>'
        '<table class="ltx_tabular"><tr><th>Model</th><th>Acc</th></tr>'
        "<tr><td>A</td><td>91.2</td></tr></table></figure>"
    )
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER), tbl))
    text = doc.sections[1].text
    assert "Table 1: Scores." in text
    assert "Model | Acc" in text and "A | 91.2" in text


def test_span_based_table_is_not_dropped(parser):
    """LaTeXML emits nested <span class=ltx_tabular> for \\resizebox tables; these carry results."""
    cell = lambda c: f'<span class="ltx_td">{c}</span>'  # noqa: E731
    row = lambda *cs: f'<span class="ltx_tr">{"".join(cell(c) for c in cs)}</span>'  # noqa: E731
    tbl = (
        '<figure class="ltx_table"><figcaption class="ltx_caption">Table 2: Deltas.</figcaption>'
        f'<span class="ltx_tabular">{row("Metric", "Value")}{row(math(r"\Delta\tau"), "0.5")}</span></figure>'
    )
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER), tbl))
    assert "Metric | Value" in doc.sections[1].text
    assert r"$\Delta\tau$ | 0.5" in doc.sections[1].text


def test_figure_keeps_caption_and_skips_images(parser):
    fig = (
        '<figure class="ltx_figure"><img src="x.png" alt="IMAGE-ALT"/>'
        '<figcaption class="ltx_caption">Figure 1: A diagram.</figcaption></figure>'
    )
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER), fig))
    text = doc.sections[1].text
    assert "Figure 1: A diagram." in text
    assert "IMAGE-ALT" not in text


def test_footnote_is_kept_inline_and_marked(parser):
    note = '<span class="ltx_note ltx_role_footnote"><span class="ltx_note_content">A side remark.</span></span>'
    doc = _sections(parser, section("S1", "1", "Intro", para("Claim", note, FILLER)))
    assert "(note: A side remark.)" in doc.sections[1].text


def test_lists_are_rendered_as_bullets(parser):
    ul = '<ul class="ltx_itemize"><li class="ltx_item">first point</li><li class="ltx_item">second point</li></ul>'
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER), ul))
    assert "- first point\n- second point" in doc.sections[1].text


# --------------------------------------------------------------- references


def test_references_keep_math_and_external_links(parser):
    refs = (
        '<li class="ltx_bibitem" id="bib.bib1"><span class="ltx_bibblock">'
        f"A. Author. On {math(r'\mathbb{P}^{n}')} spaces. "
        '<a class="ltx_bib_external" href="https://doi.org/10.1/abc">doi</a>'
        '<a href="#S1">internal</a></span></li>'
    )
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER)), refs=refs)
    [ref] = doc.references
    assert ref.key == "bib.bib1"
    assert r"$\mathbb{P}^{n}$" in ref.text
    assert ref.urls == ["https://doi.org/10.1/abc"]  # in-document anchors are not URLs


# ------------------------------------------------------------ paper shapes


def test_letter_style_body_without_sections_becomes_main_text(parser):
    """PRL-style papers put paragraphs directly in <article>; dropping them lost 60% of the math."""
    doc = _sections(
        parser,
        para("Intro paragraph", FILLER, pid="p1"),
        equation("H=E", "(1)", eid="E1"),
        para("More text", math("x^{2}"), pid="p2"),
        section("A1", "A", "Supplement", para(FILLER), kind="ltx_appendix"),
    )
    titles = [s.title for s in doc.sections]
    assert titles == ["Abstract", "Main Text", "Supplement"]
    main = doc.sections[1]
    assert "$$H=E \\tag{1}$$" in main.text and "$x^{2}$" in main.text
    assert doc.equations[0].section_path == "Main Text"


def test_abstract_is_its_own_section(parser):
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER)), abstract="We prove a thing.")
    assert doc.sections[0].title == "Abstract"
    assert doc.sections[0].text == "We prove a thing."


def test_bibliography_is_not_part_of_the_text(parser):
    refs = '<li class="ltx_bibitem" id="b1"><span class="ltx_bibblock">UNIQUE-REF-TEXT</span></li>'
    doc = _sections(parser, section("S1", "1", "Intro", para(FILLER)), refs=refs)
    assert not any("UNIQUE-REF-TEXT" in s.text for s in doc.sections)


# ------------------------------------------------------------------ errors


def test_non_arxiv_html_is_rejected(parser):
    with pytest.raises(ParseError) as exc:
        parser.parse("<html><body><p>Just a web page.</p></body></html>")
    assert exc.value.code == "PARSE_HTML_ERROR"


def test_too_short_paper_is_rejected():
    with pytest.raises(ParseError) as exc:
        HtmlParser(min_words=10_000).parse(page(section("S1", "1", "Intro", para(FILLER))))
    assert exc.value.code == "PARSE_LENGTH_ERROR"


def test_stats_count_math_and_structure(parser):
    doc = _sections(
        parser,
        section(
            "S1", "1", "Intro", para("a", math("x"), "b", math("y"), FILLER), equation("z", "(1)")
        ),
    )
    assert doc.stats["display_equations"] == 1
    assert doc.stats["inline_math"] == 2
    assert doc.stats["sections"] == 1


# ------------------------------------------------- literal dollar signs in prose


def test_a_literal_dollar_in_prose_is_escaped_so_it_cannot_open_a_math_span(parser):
    """'$100' in real papers must not pair with the next formula's '$' and corrupt the markup."""
    doc = _sections(
        parser,
        section("S1", "1", "Intro", para("API cost is $100 per run, and", math("x^{2}"), FILLER)),
    )
    text = doc.sections[1].text
    assert r"\$100 per run" in text
    assert "$x^{2}$" in text  # the real formula is still a clean span


def test_a_dollar_inside_a_formula_is_left_alone(parser):
    doc = _sections(
        parser, section("S1", "1", "Intro", para("price", math(r"\text{\$}5"), "then", FILLER))
    )
    assert r"$\text{\$}5$" in doc.sections[1].text


def test_math_inside_a_footnote_is_kept_as_latex(parser):
    note = (
        '<span class="ltx_note ltx_role_footnote"><span class="ltx_note_content">'
        f"See {math('n\geq 2')} here.</span></span>"
    )
    doc = _sections(parser, section("S1", "1", "Intro", para("Claim", note, FILLER)))
    assert r"(note: See $n\geq 2$ here.)" in doc.sections[1].text
