"""LaTeX inside the trailer's JSON strings.

Found in the browser: a tutor exercise lost its formulas. The model had written
{"question": "... $\\min(d, k)$ ..."} with a single backslash, which is not valid JSON, so the whole
trailer (verdict, next question, answer key) failed to parse. Worse, \\f \\t \\b \\r \\n ARE JSON
escapes, so "\\frac" parsed successfully as a form feed followed by "rac".
"""

import pytest

from src.agents.trailer import parse_meta, repair_json_escapes


def test_a_latex_command_with_a_single_backslash_no_longer_loses_the_trailer():
    meta = parse_meta(r'{"question": "if $r$ approaches $\min(d, k)$?", "verdict": "partial"}')
    assert meta == {"question": r"if $r$ approaches $\min(d, k)$?", "verdict": "partial"}


@pytest.mark.parametrize(
    "tex",
    [
        r"$\frac{a}{b}$",  # \f is a JSON form feed
        r"$\text{note}$",  # \t is a JSON tab
        r"$\beta + \bar{x}$",  # \b is a JSON backspace
        r"$\rho \rightarrow \right)$",  # \r is a JSON carriage return
        r"$\nu \neq \nabla$",  # \n is a JSON newline
        r"$\{a, b\}$",  # \{ is not a JSON escape at all
        r"$\Delta W = BA$",
    ],
)
def test_latex_survives_the_json_round_trip_exactly(tex):
    meta = parse_meta('{"answer_key": "' + tex + '"}')
    assert meta == {"answer_key": tex}


def test_real_json_escapes_are_still_decoded():
    meta = parse_meta(
        '{"q": "line one\\nline two", "r": "say \\"hi\\"", "s": "caf\\u00e9", "t": "a\\\\b"}'
    )
    assert meta == {"q": "line one\nline two", "r": 'say "hi"', "s": "café", "t": "a\\b"}


def test_a_correctly_escaped_formula_is_left_alone():
    assert parse_meta(r'{"q": "$\\min(d, k)$"}') == {"q": r"$\min(d, k)$"}


def test_an_escaped_newline_directly_before_a_word_is_still_a_newline():
    assert parse_meta('{"q": "first\\nNext line"}') == {"q": "first\nNext line"}


def test_repair_only_touches_backslashes():
    assert repair_json_escapes('{"a": 1, "b": "plain text"}') == '{"a": 1, "b": "plain text"}'


def test_garbage_is_still_rejected():
    assert parse_meta("not json at all") is None
    assert parse_meta('{"unterminated": ') is None


def test_a_trailing_comma_is_tolerated():
    assert parse_meta('{"question": "Why?",}') == {"question": "Why?"}
    assert parse_meta('{"a": [1, 2,], "b": 3}') == {"a": [1, 2], "b": 3}
