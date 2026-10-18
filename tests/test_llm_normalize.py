"""Model-specific citation styles are rewritten to [n], including across stream chunks."""

import pytest

from src.agents.grounding import cited_numbers
from src.llm.normalize import OutputGuard, normalize_text


def test_fullwidth_markers_become_ascii_brackets():
    assert normalize_text("A fact 【2】 and another【1】.") == "A fact [2] and another [1]."


def test_line_references_are_dropped():
    assert normalize_text("Pruned blocks【1†L10-L14】.") == "Pruned blocks [1]."


def test_plain_text_and_real_daggers_are_untouched():
    text = "Ordinary [3] cite, and a footnote† marker."
    assert normalize_text(text) == text


@pytest.mark.parametrize("cut", range(1, 12))
def test_a_marker_split_across_chunks_is_still_rewritten(cut):
    text = "Speedup【12†L3-L9】 holds."
    norm = OutputGuard(hold=0)
    streamed = norm.feed(text[:cut]) + norm.feed(text[cut:])
    assert streamed == "Speedup [12] holds."


def test_normalised_markers_count_as_citations_downstream():
    assert cited_numbers(normalize_text("It prunes blocks【2】 and scales【1】.")) == [2, 1]


# ---------------------------------------------------------------------------- whitespace guard

import pytest  # noqa: E402

from src.llm.normalize import DegenerateOutput  # noqa: E402


def _stream(guard: OutputGuard, chunks: list[str]) -> str:
    return "".join(guard.feed(c) for c in chunks) + guard.flush()


def test_padded_table_cells_are_collapsed():
    padded = "| Paper" + " " * 40 + "| Approach" + " " * 60 + "|\n|---|---|\n| A | B |\n"
    assert normalize_text(padded) == "| Paper    | Approach    |\n|---|---|\n| A | B |\n"


def test_leading_indentation_is_preserved_for_nested_lists_and_code():
    text = "- item\n    - nested\n        deeper\n"
    assert normalize_text(text) == text


def test_a_runaway_run_of_spaces_is_degenerate_output():
    """The real failure: 16,103 spaces and 29 other characters, whole token budget gone."""
    with pytest.raises(DegenerateOutput):
        normalize_text("## Comparison\n\n| Paper" + " " * 500 + "| Approach")


def test_degenerate_start_is_detected_before_anything_is_emitted():
    """Held back, so the router can still fail over to another provider (it cannot after a token)."""
    guard = OutputGuard()
    emitted = guard.feed("## Comparison\n\n| Paper")
    assert emitted == ""  # still inside the hold-back window
    with pytest.raises(DegenerateOutput):
        for _ in range(10):
            emitted += guard.feed(" " * 40)
    assert emitted == ""


@pytest.mark.parametrize("size", [1, 3, 7, 50])
def test_chunking_never_changes_the_result(size):
    text = (
        "Pruned blocks【1†L3-L9】 give speed"
        + " " * 30
        + "ups.\n\n- a\n    - b 【2】\n"
        + "x" * 200
    )
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    assert _stream(OutputGuard(), chunks) == normalize_text(text)


def test_short_answers_are_flushed_at_the_end():
    guard = OutputGuard()
    assert guard.feed("Short.") == ""
    assert guard.flush() == "Short."
