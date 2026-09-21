"""Reading-order recovery. Page width is 600; a column is ~250 wide, gutter around x=300."""

import pymupdf
import pytest

from src.pipeline.layout import Block, is_two_column, order_blocks
from src.pipeline.parser import PDFParser

W = 600.0


def blk(label: str, x0: float, y0: float, x1: float, y1: float | None = None) -> Block:
    text = f"{label} " + "lorem ipsum dolor sit amet " * 4  # long enough to count as body text
    return Block(x0, y0, x1, y1 if y1 is not None else y0 + 40, text)


def L(label, y):  # left column paragraph
    return blk(label, 50, y, 290)


def R(label, y):  # right column paragraph
    return blk(label, 310, y, 550)


def FULL(label, y):  # full-width element (title, abstract, wide figure)
    return blk(label, 50, y, 550)


def frag(label: str, x0: float, y: float) -> Block:
    """A tiny piece of an equation, as PyMuPDF shatters them."""
    return Block(x0, y, x0 + 25, y + 10, label)


def labels(blocks):
    return [b.text.split()[0] for b in blocks]


# ------------------------------------------------------- pages left untouched


def test_single_column_page_is_left_alone():
    blocks = [FULL("a", 50), FULL("b", 120), FULL("c", 190)]
    assert order_blocks(blocks, W) == blocks


def test_two_column_page_already_in_reading_order_is_left_alone():
    """PyMuPDF's stream order is usually right; 'fixing' it can only make things worse."""
    blocks = [L("l1", 50), L("l2", 120), R("r1", 50), R("r2", 120)]
    assert order_blocks(blocks, W) == blocks


def test_equation_fragments_are_never_mistaken_for_columns():
    """The real failure: single-column pages full of display math looked 'two-column' and got shredded."""
    blocks = [
        FULL("intro", 50),
        frag("f1", 150, 120),
        frag("f2", 175, 118),
        frag("f3", 330, 120),
        frag("f4", 360, 119),
        frag("f5", 200, 140),
        frag("f6", 340, 141),
        FULL("outro", 200),
    ]
    assert order_blocks(blocks, W) == blocks


def test_one_block_per_side_is_not_enough_to_call_it_two_columns():
    blocks = [L("l1", 50), R("r1", 50)]
    assert not is_two_column(blocks, W)


# ------------------------------------------------------------ pages repaired


def test_interleaved_columns_are_read_left_then_right():
    blocks = [L("l1", 50), R("r1", 50), L("l2", 120), R("r2", 120)]
    assert labels(order_blocks(blocks, W)) == ["l1", "l2", "r1", "r2"]


def test_full_width_title_stays_first_then_columns_in_order():
    blocks = [FULL("title", 20), L("l1", 90), R("r1", 90), L("l2", 160), R("r2", 160)]
    assert labels(order_blocks(blocks, W)) == ["title", "l1", "l2", "r1", "r2"]


def test_full_width_figure_splits_the_page_into_bands():
    blocks = [
        L("l1", 50),
        R("r1", 50),
        L("l2", 100),
        R("r2", 100),
        FULL("figure", 300),
        L("l3", 400),
        R("r3", 400),
        L("l4", 450),
        R("r4", 450),
    ]
    assert labels(order_blocks(blocks, W)) == [
        "l1",
        "l2",
        "r1",
        "r2",
        "figure",
        "l3",
        "l4",
        "r3",
        "r4",
    ]


def test_small_fragments_stay_attached_to_the_block_they_follow():
    """Reordering columns must not tear an equation away from its paragraph or scramble it."""
    blocks = [
        L("l1", 50),
        frag("eqA", 100, 95),
        frag("eqB", 130, 96),
        R("r1", 50),
        L("l2", 120),
        R("r2", 120),
    ]
    assert labels(order_blocks(blocks, W)) == ["l1", "eqA", "eqB", "l2", "r1", "r2"]


def test_input_list_is_not_mutated():
    blocks = [L("l1", 50), R("r1", 50), L("l2", 120), R("r2", 120)]
    snapshot = list(blocks)
    order_blocks(blocks, W)
    assert blocks == snapshot


# ------------------------------------------------- against a real PDF (PyMuPDF)


@pytest.fixture
def interleaved_pdf_page(tmp_path):
    """A real two-column PDF whose content stream alternates left/right paragraphs."""
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=800)

    def put(rect, marker):
        body = f"{marker} " + "the quick brown fox jumps over the lazy dog " * 6
        assert page.insert_textbox(pymupdf.Rect(*rect), body, fontsize=9) > 0

    # written in an interleaved order: left1, right1, left2, right2
    put((50, 50, 290, 150), "LEFTONE")
    put((310, 50, 550, 150), "RIGHTONE")
    put((50, 300, 290, 400), "LEFTTWO")
    put((310, 300, 550, 400), "RIGHTTWO")
    path = tmp_path / "twocol.pdf"
    doc.save(str(path))
    doc.close()
    return pymupdf.open(str(path))[0]


def test_pdf_page_text_is_read_column_by_column(interleaved_pdf_page):
    text = PDFParser._page_text(interleaved_pdf_page)
    order = [text.index(m) for m in ("LEFTONE", "LEFTTWO", "RIGHTONE", "RIGHTTWO")]
    assert order == sorted(order), text[:400]
