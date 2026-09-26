"""Deterministic offline "LLM" - lets the whole product run with zero API keys.

It is extractive, not generative: it quotes the retrieved sources it was given (lines starting with
"[n]") and cites them, so a demo is still grounded and the guardrail/citation pipeline is exercised
end to end. Also used by CI so tests never spend money or depend on the network.
"""

import json
import re
from collections.abc import AsyncIterator

from src.llm.base import LLMRequest, LLMResult, Usage

_SOURCE = re.compile(r"^\[(\d+)\]\s*(.+)$", re.MULTILINE)


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
        if not sources:
            return (
                "(Demo mode: no LLM provider is configured.) I couldn't find material on this "
                "in the indexed papers. Try ingesting papers on the topic, or rephrase."
            )
        lines = ["(Demo mode: extractive answer from the retrieved sources.)", ""]
        for num, body in sources[:3]:
            sentence = re.split(r"(?<=[.!?])\s", body.strip(), maxsplit=1)[0]
            lines.append(f"- {sentence[:280]} [{num}]")
        if request.task == "tutor":
            lines += ["", "Which part of this would you like to go through step by step?"]
        return "\n".join(lines)

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
