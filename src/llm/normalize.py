"""Provider output quirks, normalised once at the LLM boundary.

Two real quirks found by running the agents against live models:

* Citations. Some models (notably gpt-oss on Groq) cite as `【1】` or `【1†L10-L14】` instead of
  the `[1]` the prompts ask for. Everything downstream (citation linking, the grounding checks,
  the sidenotes) speaks `[n]`, so a model's habit must not decide whether an answer is cited.
* Whitespace padding. Gemini sometimes aligns a Markdown table by padding cells with spaces and
  never stops: one answer was 16,103 spaces and 29 other characters, and spent the whole token
  budget before writing a single row of content.

Streaming makes both stateful: a marker can be split across chunks (`【`, `1`, `†L3`, `】`) and a
space run can span many of them, so the guard carries state between chunks rather than matching a
regex on each one. It also holds back the first few characters, so that a degenerate start is
detected BEFORE anything is shown, while the router can still fail over to another provider.
"""

HOLD_CHARS = 120  # held back before the first emission, so a bad start can still fail over
MAX_SPACE_RUN = 4  # mid-line runs are collapsed to this (markdown ignores them anyway)
MAX_INDENT = 16  # leading indentation (nested lists, code) is allowed to be wider
DEGENERATE_RUN = 200  # this many spaces in a row is the model looping, not formatting


class DegenerateOutput(ValueError):
    """The model is emitting padding instead of content."""


class OutputGuard:
    """Per-answer filter: citation style, whitespace runs, degenerate output. Not thread-safe."""

    def __init__(self, hold: int = HOLD_CHARS) -> None:
        self._hold = hold
        self._held = ""  # normalised text not yet released
        self._released = False  # the hold-back buffer has been flushed once
        self._in_marker = False  # inside a 【...】 marker we rewrote
        self._in_tail = False  # after a "†" inside such a marker: drop until the close
        self._last = ""  # last character emitted by the citation stage
        self._run = 0  # consecutive spaces seen (including dropped ones)
        self._at_line_start = True

    # --- stage 1: citations ------------------------------------------------------------------
    def _citations(self, chunk: str) -> str:
        out: list[str] = []
        for ch in chunk:
            if ch == "【":
                self._in_marker, self._in_tail = True, False
                # The downstream citation pattern needs a non-word character before "[" (so that
                # array indexing like x[1] is not read as a citation), and these models attach
                # the marker to the word: "baselines【2】".
                if (out[-1] if out else self._last).isalnum():
                    out.append(" ")
                out.append("[")
            elif ch == "】":
                self._in_marker = self._in_tail = False
                out.append("]")
            elif self._in_marker and ch == "†":
                self._in_tail = True  # "†L3-L9" is a line reference, not part of the number
            elif self._in_tail:
                if ch == "]":  # a model that mixed bracket styles
                    self._in_marker = self._in_tail = False
                    out.append("]")
            else:
                out.append(ch)
        result = "".join(out)
        if result:
            self._last = result[-1]
        return result

    # --- stage 2: whitespace -----------------------------------------------------------------
    def _whitespace(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            if ch == " ":
                self._run += 1
                if self._run >= DEGENERATE_RUN:
                    raise DegenerateOutput(f"{self._run}+ consecutive spaces")
                limit = MAX_INDENT if self._at_line_start else MAX_SPACE_RUN
                if self._run <= limit:
                    out.append(ch)
                continue
            self._run = 0
            self._at_line_start = ch == "\n"
            out.append(ch)
        return "".join(out)

    # --- public --------------------------------------------------------------------------------
    def feed(self, chunk: str) -> str:
        """Text that is safe to show now. May be empty while the start is being held back."""
        clean = self._whitespace(self._citations(chunk))
        if self._released:
            return clean
        self._held += clean
        if len(self._held) < self._hold:
            return ""
        self._released, released, self._held = True, self._held, ""
        return released

    def flush(self) -> str:
        """The held-back remainder, at the end of a stream."""
        self._released, rest, self._held = True, self._held, ""
        return rest


def normalize_text(text: str) -> str:
    """The same filter for a complete (non-streamed) answer; raises DegenerateOutput."""
    guard = OutputGuard()
    return guard.feed(text) + guard.flush()
