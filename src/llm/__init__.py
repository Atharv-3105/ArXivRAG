"""LLM layer: provider adapters behind a failover router (see router.py)."""

from src.llm.base import ChatMessage, ErrorKind, LLMRequest, LLMResult, ProviderError, Usage
from src.llm.router import (
    LLMRefusedError,
    LLMRouter,
    LLMStreamInterruptedError,
    LLMUnavailableError,
    StreamEvent,
)

__all__ = [
    "ChatMessage",
    "ErrorKind",
    "LLMRefusedError",
    "LLMRequest",
    "LLMResult",
    "LLMRouter",
    "LLMStreamInterruptedError",
    "LLMUnavailableError",
    "ProviderError",
    "StreamEvent",
    "Usage",
]
