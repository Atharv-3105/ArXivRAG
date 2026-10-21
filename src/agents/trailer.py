"""One LLM call, two outputs: prose for the reader (streamed) + a JSON trailer for the machine.

The tutor needs structured fields (the check question, its answer key, the grade) *and* a streamed
reply. JSON mode would kill streaming; a second call would double quota and latency. So the prompt
asks for the reply, then a line `<<<META>>>`, then one line of JSON. The splitter streams
everything before the delimiter and swallows the rest - holding back only as many characters as
could be the start of a delimiter split across two tokens.

If a model ignores the format (no delimiter, invalid JSON), the reply is still shown and the
callers repair what they can from the prose (see checks.last_question) - the turn never fails
because of the trailer.
"""

import json
import re

DELIMITER = "<<<META>>>"


class TrailerSplitter:
    def __init__(self, delimiter: str = DELIMITER) -> None:
        self.delimiter = delimiter
        self._held = ""
        self._meta: list[str] = []
        self.in_meta = False

    def _held_back(self, text: str) -> int:
        """Length of the longest suffix of `text` that is a prefix of the delimiter."""
        for size in range(min(len(text), len(self.delimiter) - 1), 0, -1):
            if self.delimiter.startswith(text[-size:]):
                return size
        return 0

    def feed(self, chunk: str) -> str:
        """Returns the part of `chunk` that is safe to show now."""
        if self.in_meta:
            self._meta.append(chunk)
            return ""
        text = self._held + chunk
        index = text.find(self.delimiter)
        if index >= 0:
            self.in_meta = True
            self._meta.append(text[index + len(self.delimiter) :])
            self._held = ""
            return text[:index]
        keep = self._held_back(text)
        self._held = text[len(text) - keep :] if keep else ""
        return text[: len(text) - keep]

    def close(self) -> str:
        """Flush what was held back (it was not a delimiter after all)."""
        rest, self._held = self._held, ""
        return "" if self.in_meta else rest

    @property
    def meta_text(self) -> str:
        return "".join(self._meta)


# LaTeX commands a model writes inside a JSON string ("question": "if $\min(d, k)$ ..."). JSON has
# no \m, so the whole trailer used to fail to parse and the turn lost its verdict, next question and
# answer key. Worse, \f \t \b \r \n ARE JSON escapes: "\frac" parsed fine as a form feed + "rac".
_LATEX = frozenset(
    """frac dfrac tfrac text textbf textit textrm mathbf mathrm mathcal mathbb mathit mathsf min max
    sum prod int oint lim log ln exp sin cos tan sec alpha beta gamma delta epsilon varepsilon zeta
    eta theta vartheta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi varphi chi psi omega
    Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega ll gg le leq ge geq ne neq approx sim simeq
    equiv propto cdot times div pm mp in notin ni subset subseteq supset supseteq cup cap setminus
    emptyset to rightarrow leftarrow Rightarrow Leftarrow leftrightarrow mapsto nabla partial infty
    sqrt left right big Big bigg Bigg bar hat tilde vec dot ddot ldots cdots dots quad qquad begin
    end operatorname argmin argmax top bot forall exists neg land lor langle rangle lceil rceil
    lfloor rfloor mid parallel perp circ star ast bullet oplus otimes odot underbrace overbrace
    underline overline boldsymbol mathop limits displaystyle tag label""".split()
)
_WORD = re.compile(r"[A-Za-z]+")
_HEX4 = re.compile(r"[0-9a-fA-F]{4}")
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def repair_json_escapes(text: str) -> str:
    """Double the backslash of every LaTeX command or other invalid escape inside the text, and
    leave real JSON escapes (\\n, \\", \\\\, \\u00e9, ...) alone."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        if nxt in ("\\", '"', "/"):  # a valid pair: keep both characters
            out.append(ch + nxt)
            i += 2
            continue
        if nxt == "u" and _HEX4.fullmatch(text[i + 2 : i + 6]):
            out.append(text[i : i + 6])
            i += 6
            continue
        word = _WORD.match(text, i + 1)
        if word and word.group() not in _LATEX and word.group()[0] in "bfnrt":
            out.append(ch)  # \n, \t ... followed by ordinary text: a genuine escape
        else:
            out.append("\\\\")  # a LaTeX command, or an escape JSON does not have
        i += 1
    return "".join(out)


def parse_meta(text: str) -> dict | None:
    """The JSON object in a trailer, tolerating code fences, chatter around it, and LaTeX."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = repair_json_escapes(text[start : end + 1])
    try:
        value = json.loads(candidate)
    except ValueError:
        try:  # models sometimes leave a trailing comma: {"question": "Why?",}
            value = json.loads(_TRAILING_COMMA.sub(r"\1", candidate))
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def split_complete(text: str) -> tuple[str, dict | None]:
    """Non-streaming convenience: (visible text, meta)."""
    splitter = TrailerSplitter()
    visible = splitter.feed(text) + splitter.close()
    return visible.rstrip(), parse_meta(splitter.meta_text) if splitter.in_meta else None
