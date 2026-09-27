"""Deterministic offline "LLM" - lets the whole product run with zero API keys.

It is extractive, not generative: it quotes the retrieved sources it was given (lines starting with
"[n]") and cites them, so a demo is still grounded and the guardrail/citation pipeline is exercised
end to end. It follows each specialist's output contract (request.tags["agent"]): the tutor's
<<<META>>> trailer, the critic's and synthesizer's sections, and it grades tutor answers by word
overlap with the answer key. Also used by CI so tests never spend money or depend on the network.
"""

import json
import re
from collections.abc import AsyncIterator

from src.llm.base import LLMRequest, LLMResult, Usage

_SOURCE = re.compile(r"^\[(\d+)\]\s*(.+)$", re.MULTILINE)
_WORD = re.compile(r"[a-z][a-z0-9\-]{3,}")
DELIMITER = "<<<META>>>"


def _first_sentence(body: str, limit: int = 240) -> str:
    sentence = re.split(r"(?<=[.!])\s", body.strip(), maxsplit=1)[0].replace("?", ".")
    if len(sentence) <= limit:
        return sentence
    cut = sentence[:limit].rsplit(" ", 1)[0]
    if len(re.findall(r"(?<!\\)\$", cut)) % 2:  # never end inside a $...$ formula
        cut = cut[: cut.rfind("$")].rstrip()
    return cut + " ..."


def _field(prompt: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}:\s*(.*)$", prompt, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _overlap(answer: str, key: str) -> float:
    a, k = set(_WORD.findall(answer.lower())), set(_WORD.findall(key.lower()))
    return len(a & k) / len(k) if k else 0.0


class MockAdapter:
    def __init__(self, name: str = "mock", model: str = "mock-extractive-1"):
        self.name = name
        self.model = model

    async def aclose(self) -> None:
        return None

    def _answer(self, request: LLMRequest) -> str:
        prompt = "\n".join(m.content for m in request.messages)
        if request.json_mode:
            return json.dumps({"intent": "chat", "confidence": 0.5, "concepts": []})
        sources = _SOURCE.findall(prompt)
        agent = request.tags.get("agent")
        if agent == "tutor":
            return self._tutor(request, prompt, sources)
        if not sources:
            return (
                "(Demo mode: no LLM provider is configured.) I couldn't find material on this "
                "in the indexed papers. Try ingesting papers on the topic, or rephrase."
            )
        if agent == "critic":
            return self._critic(sources)
        if agent == "synthesizer":
            return self._synthesizer(sources)
        lines = ["(Demo mode: extractive answer from the retrieved sources.)", ""]
        for num, body in sources[:3]:
            lines.append(f"- {_first_sentence(body)} [{num}]")
        return "\n".join(lines)

    # ------------------------------------------------------------ specialists

    def _tutor(self, request: LLMRequest, prompt: str, sources) -> str:
        action = request.tags.get("action", "explain")
        question = _field(prompt, "Question")
        key = _field(prompt, "Answer key")
        if action == "grade":
            answer = request.messages[-1].content
            score = _overlap(answer, key)
            verdict = "correct" if score >= 0.5 else "partial" if score >= 0.2 else "incorrect"
            if verdict == "correct":
                text = "(Demo mode) That's right: your answer matches the key idea."
                meta = {"verdict": verdict, "question": None, "answer_key": None}
            else:
                hint = " ".join(key.split()[:5])
                text = f"(Demo mode) Not quite. Hint: think about {hint} ... {question}"
                meta = {"verdict": verdict, "question": question, "answer_key": key}
            return f"{text}\n{DELIMITER}\n{json.dumps(meta)}"
        if action == "hint":
            hint = " ".join(key.split()[:6])
            text = f"(Demo mode) Hint: start from {hint} ... {question}"
            return f"{text}\n{DELIMITER}\n{json.dumps({'question': question})}"
        lines = ["(Demo mode: extractive explanation from the sources.)", ""]
        for num, body in sources[:2]:
            lines.append(f"- {_first_sentence(body)} [{num}]")
        check = "In your own words, what is the key idea of source [1]?"
        answer_key = _first_sentence(sources[0][1]) if sources else ""
        lines += ["", check]
        meta = {"question": check, "answer_key": answer_key, "concept": None}
        return "\n".join(lines) + f"\n{DELIMITER}\n{json.dumps(meta)}"

    @staticmethod
    def _critic(sources) -> str:
        cite = [f"{_first_sentence(b, 160)} [{n}]" for n, b in sources[:3]]
        return "\n".join(
            [
                "## Summary",
                f"(Demo mode) {cite[0]}",
                "## Strengths",
                *(f"- {c}" for c in cite[:2]),
                "## Weaknesses",
                f"- The sources do not report variance across seeds [{sources[0][0]}]",
                "## Assumptions",
                "- Stated in the sources as quoted above.",
                "## Baselines and evaluation",
                "- See the cited sections.",
                "## Reproducibility",
                "- Not settled by the sources.",
                "## Open questions",
                "- Whether the results transfer to other settings.",
            ]
        )

    @staticmethod
    def _synthesizer(sources) -> str:
        rows = [
            f"| source [{n}] | {_first_sentence(b, 80)} | see [{n}] | as reported |"
            for n, b in sources[:4]
        ]
        return "\n".join(
            [
                "## Comparison",
                "| Paper | Approach | Key result | Setting |",
                "|---|---|---|---|",
                *rows,
                "## Where they agree",
                f"- (Demo mode) Both address the question [{sources[0][0]}]",
                "## Where they disagree",
                "- Not settled by the retrieved passages.",
                "## Timeline",
                "- In publication order of the sources.",
                "## Open directions",
                "- Direct comparison under one setup.",
            ]
        )

    # ------------------------------------------------------------ adapter protocol

    async def complete(self, request: LLMRequest) -> LLMResult:
        text = self._answer(request)
        return LLMResult(
            text, self.name, self.model, Usage(request.estimated_prompt_tokens(), len(text) // 4)
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str | Usage]:
        text = self._answer(request)
        for token in re.findall(r"\S+\s*", text):
            yield token
        yield Usage(request.estimated_prompt_tokens(), len(text) // 4)
