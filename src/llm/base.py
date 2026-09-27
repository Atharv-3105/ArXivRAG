"""Provider-neutral contract between the app and any LLM.

Adapters translate one provider's API into this contract; the router (router.py) only ever sees
these types. That keeps provider quirks (header names, error bodies, SSE formats) out of the routing
logic, and lets tests drive the router with a fake adapter.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass
class LLMRequest:
    messages: list[ChatMessage]
    task: str = "default"  # selects the provider order (settings.LLM_TASK_ORDERS)
    max_tokens: int = 1500
    temperature: float = 0.2
    json_mode: bool = False  # ask for a JSON object (classification, extraction)
    # Free-form labels (agent, action, prompt version). Adapters ignore them; logs, traces and
    # the offline mock use them.
    tags: dict = field(default_factory=dict)

    def estimated_prompt_tokens(self) -> int:
        """~4 characters per token. Only used for budgeting, never billing."""
        return sum(len(m.content) for m in self.messages) // 4 + 1


@dataclass
class Usage:
    """Token usage as reported by the provider (None when it did not say)."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


class ErrorKind(Enum):
    """How a failure should affect routing. Decided from status codes / exception types."""

    RATE_LIMITED = "rate_limited"  # 429: cool down for Retry-After, no breaker penalty
    TRANSIENT = "transient"  # 5xx, timeout, network: counts toward the circuit breaker
    AUTH = "auth"  # 401/403: misconfigured key - open the breaker immediately
    BAD_REQUEST = "bad_request"  # 400/404/422: this provider rejects the request shape
    EMPTY = "empty"  # 200 but no usable text
    INVALID_OUTPUT = "invalid_output"  # text failed the caller's validation
    REFUSAL = "refusal"  # safety decline: content-specific, not the provider's health


class ProviderError(Exception):
    def __init__(
        self,
        provider: str,
        kind: ErrorKind,
        message: str,
        retry_after: float | None = None,
        status_code: int | None = None,
    ):
        super().__init__(f"[{provider}] {kind.value}: {message}")
        self.provider = provider
        self.kind = kind
        self.message = message
        self.retry_after = retry_after
        self.status_code = status_code


def classify_status(status: int) -> ErrorKind:
    if status == 429:
        return ErrorKind.RATE_LIMITED
    if status in (401, 403):
        return ErrorKind.AUTH
    if status in (408, 409) or status >= 500:
        return ErrorKind.TRANSIENT
    return ErrorKind.BAD_REQUEST


def parse_retry_after(headers: dict[str, str] | None, default: float) -> float:
    """Seconds to wait, from standard or provider-specific headers.

    `Retry-After: 30` is standard. Groq/OpenAI also send `x-ratelimit-reset-requests: 2m59.5s`.
    Never guessed from free text (a number in an error message can be anything).
    """
    if not headers:
        return default
    lower = {k.lower(): v for k, v in headers.items()}
    raw = lower.get("retry-after")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass  # HTTP-date form: fall through to the default
    for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        if key in lower:
            parsed = _parse_duration(lower[key])
            if parsed is not None:
                return parsed
    return default


def _parse_duration(value: str) -> float | None:
    """'2m59.56s' / '7.66s' / '120ms' -> seconds."""
    import re

    total, matched = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", value.strip()):
        matched = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if matched else None


@runtime_checkable
class LLMAdapter(Protocol):
    """One configured model on one provider."""

    name: str  # unique slot name, e.g. "groq", "openrouter:z-ai/glm-5.2:free"
    model: str

    async def complete(self, request: LLMRequest) -> LLMResult: ...

    def stream(self, request: LLMRequest) -> AsyncIterator[str | Usage]:
        """Yield text deltas; may yield one Usage object at the end."""
        ...

    async def aclose(self) -> None: ...
