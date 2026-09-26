"""The contract between graph nodes. Each node reads some keys and returns a partial update."""

import operator
from typing import Annotated, Literal, TypedDict

Intent = Literal["learn", "critique", "synthesize", "recommend", "chat"]
INTENTS: tuple[Intent, ...] = ("learn", "critique", "synthesize", "recommend", "chat")


class Source(TypedDict):
    n: int  # the [n] the model cites
    chunk_id: str
    paper_id: str
    title: str
    section: str
    content: str
    score: float
    url: str  # deep link: arxiv.org/html/<id>#<anchor> when the section anchor is known
    math_fidelity: str


class Citation(TypedDict):
    n: int
    paper_id: str
    title: str
    section: str
    url: str


class AgentState(TypedDict, total=False):
    # input
    query: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}], oldest first
    # classify
    intent: Intent
    intent_confidence: float
    # retrieve
    retrieval_query: str
    sources: list[Source]
    degraded: str | None  # "retrieval_unavailable" | "no_sources": answered without the LLM
    # generate
    response: str
    citations: list[Citation]
    ungrounded: list[int]  # [n] markers that point at no retrieved source
    provider: str
    model: str
    # observability: every node adds its duration (merged, not overwritten)
    timings: Annotated[dict[str, float], operator.or_]
