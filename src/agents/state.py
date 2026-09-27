"""The contract between graph nodes. Each node reads some keys and returns a partial update."""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

Intent = Literal["learn", "critique", "synthesize", "recommend", "chat"]
INTENTS: tuple[Intent, ...] = ("learn", "critique", "synthesize", "recommend", "chat")
Agent = Literal["tutor", "critic", "synthesizer", "advisor", "general"]
AGENTS: tuple[Agent, ...] = ("tutor", "critic", "synthesizer", "advisor", "general")


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
    role: str  # retrieved | prerequisite | related | citation_context | reading | recommendation
    published: NotRequired[str | None]


class Citation(TypedDict):
    n: int
    paper_id: str
    title: str
    section: str
    url: str


class AgentState(TypedDict, total=False):
    # input (built by the chat route)
    query: str
    user_id: int | None
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}], oldest first
    profile: dict  # expertise_level, math_depth, goals, interests
    mastery: dict[str, float]  # concept -> effective mastery, for concepts in this query
    weak_concepts: list[str]  # all of the reader's concepts below MASTERY_WEAK_BELOW
    due_reviews: list[str]  # concepts due for spaced-repetition review
    concepts: list[str]  # concepts the query is about (vocabulary slugs)
    tutor_state: dict | None  # the conversation's tutoring session (see agents/tutor.py)
    action: str | None  # explicit UI action (answer, hint, dont_understand, skip, ...)
    action_args: dict  # e.g. {"paper_id": ..., "equation": "3"}
    # route
    intent: Intent
    intent_confidence: float
    agent: Agent
    tutor_action: str | None
    plan: dict  # retrieval plan: kind (vector|none|reading|equation), query, top_k, ...
    # retrieve
    retrieval_query: str
    sources: list[Source]
    degraded: str | None  # "retrieval_unavailable" | "no_sources" | "not_found"
    graph_status: str  # "ok" | "failed" (fell back to vector-only) | "off"
    reading: dict | None  # guided reading: paper, section, index, total, prerequisites
    equation: dict | None  # equation walk-through: latex, number, section
    # specialist
    response: str
    citations: list[Citation]
    ungrounded: list[int]  # [n] markers that point at no retrieved source
    provider: str
    model: str
    prompt_version: str | None
    checks: dict[str, bool]
    verdict: str | None  # tutor grading: correct | partial | incorrect
    learning_events: list[dict]  # memory events for the chat route to submit
    # observability: every node adds its duration (merged, not overwritten)
    timings: Annotated[dict[str, float], operator.or_]
