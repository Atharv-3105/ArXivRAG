"""Reading-order recovery for PDF pages (pure functions, no PyMuPDF dependency).

PyMuPDF returns text blocks in PDF content-stream order. For most LaTeX papers that is already the
correct reading order (measured on real arXiv papers: it matches or beats every alternative), and
`sort=True` is worse because it interleaves columns line by line. So the rule here is conservative:

  * judge column structure only from substantial *text* blocks; equations are shattered into
    many tiny blocks and must never be mistaken for columns;
  * if the stream order already agrees with the column structure, return it untouched;
  * only when the stream interleaves the columns, reorder: left column then right column inside
    each horizontal band delimited by full-width blocks. Small fragments stay attached to the
    block they follow, so an equation keeps its internal order.
"""

from dataclasses import dataclass

MIN_TEXT_WIDTH_FRAC = 0.30  # a column-text block is at least this wide (columns are ~0.42 wide)
MIN_TEXT_CHARS = 60  # ...and at least about one full line of text
MIN_BLOCKS_PER_COLUMN = 2


@dataclass(frozen=True)
class Block:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def is_text_block(b: Block, page_width: float) -> bool:
    """Substantial paragraph-like block, as opposed to an equation fragment or a page number."""
    wide_enough = (b.x1 - b.x0) >= MIN_TEXT_WIDTH_FRAC * page_width
    return wide_enough and len(b.text.strip()) >= MIN_TEXT_CHARS


def is_two_column(blocks: list[Block], page_width: float) -> bool:
    """Enough substantial blocks entirely on each side of the centre, sitting side by side."""
    mid, tol = page_width / 2, page_width * 0.03
    left = [b for b in blocks if b.x1 <= mid + tol]
    right = [b for b in blocks if b.x0 >= mid - tol]
    if len(left) < MIN_BLOCKS_PER_COLUMN or len(right) < MIN_BLOCKS_PER_COLUMN:
        return False
    return any(lb.y0 < rb.y1 and rb.y0 < lb.y1 for lb in left for rb in right)


def _column_order(anchors: list[Block], page_width: float) -> list[Block]:
    mid, tol = page_width / 2, page_width * 0.03
    spanning = sorted(
        (b for b in anchors if b.x0 < mid - tol and b.x1 > mid + tol), key=lambda b: b.y0
    )
    span_ids = {id(b) for b in spanning}
    columns = [b for b in anchors if id(b) not in span_ids]

    def zone(b: Block) -> int:  # number of full-width blocks above this one
        return sum(1 for s in spanning if s.yc < b.yc)

    ordered: list[Block] = []
    for z in range(len(spanning) + 1):
        if z > 0:
            ordered.append(spanning[z - 1])
        in_zone = [b for b in columns if zone(b) == z]
        ordered += sorted((b for b in in_zone if (b.x0 + b.x1) / 2 < mid), key=lambda b: b.y0)
        ordered += sorted((b for b in in_zone if (b.x0 + b.x1) / 2 >= mid), key=lambda b: b.y0)
    return ordered


def order_blocks(blocks: list[Block], page_width: float) -> list[Block]:
    """Blocks in reading order; the input order is kept unless it is demonstrably broken."""
    anchors = [b for b in blocks if is_text_block(b, page_width)]
    if not is_two_column(anchors, page_width):
        return list(blocks)

    # Group each substantial block with the small fragments that follow it in the stream.
    prefix: list[Block] = []
    groups: list[list[Block]] = []
    for b in blocks:
        if is_text_block(b, page_width):
            groups.append([b])
        elif groups:
            groups[-1].append(b)
        else:
            prefix.append(b)

    stream_anchors = [g[0] for g in groups]
    wanted = _column_order(stream_anchors, page_width)
    if [id(b) for b in wanted] == [id(b) for b in stream_anchors]:
        return list(blocks)  # the stream already reads column by column

    by_id = {id(g[0]): g for g in groups}
    return prefix + [b for a in wanted for b in by_id[id(a)]]
