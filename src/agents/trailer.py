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


def parse_meta(text: str) -> dict | None:
    """The JSON object in a trailer, tolerating code fences and chatter around it."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def split_complete(text: str) -> tuple[str, dict | None]:
    """Non-streaming convenience: (visible text, meta)."""
    splitter = TrailerSplitter()
    visible = splitter.feed(text) + splitter.close()
    return visible.rstrip(), parse_meta(splitter.meta_text) if splitter.in_meta else None
