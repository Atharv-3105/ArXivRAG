"""Adapter for Claude via the official Anthropic SDK (a paid provider; off unless a key is set)."""

import time
from collections.abc import AsyncIterator

import anthropic

from src.llm.base import (
    ChatMessage,
    ErrorKind,
    LLMRequest,
    LLMResult,
    ProviderError,
    Usage,
    classify_status,
    parse_retry_after,
)

# Server-side refusal fallbacks ("default" routing) - supported on these model families.
_FALLBACK_FAMILIES = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicAdapter:
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        timeout: float = 60.0,
        server_fallbacks: bool = True,
        default_retry_after: float = 60.0,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.name = name
        self.model = model
        self.default_retry_after = default_retry_after
        self.server_fallbacks = server_fallbacks and model.startswith(_FALLBACK_FAMILIES)
        # max_retries=0: the router owns retry/failover. SDK-level retries would multiply latency
        # and hide 429s from the router's rate-limit bookkeeping.
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=0
        )

    async def aclose(self) -> None:
        await self._client.close()

    def _kwargs(self, request: LLMRequest) -> dict:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        convo = [m for m in request.messages if m.role != "system"]
        if not convo or convo[0].role != "user":
            convo = [ChatMessage("user", "(continue)"), *convo]
        kwargs: dict = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in convo],
        }
        # No temperature: current Claude models reject sampling parameters.
        if system:
            kwargs["system"] = system
        if request.json_mode:
            kwargs["system"] = (system + "\n\n" if system else "") + (
                "Respond with a single JSON object and nothing else."
            )
        if self.server_fallbacks:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _map_error(self, e: Exception) -> ProviderError:
        if isinstance(e, anthropic.RateLimitError):
            retry = parse_retry_after(dict(e.response.headers), self.default_retry_after)
            return ProviderError(self.name, ErrorKind.RATE_LIMITED, str(e), retry, 429)
        if isinstance(e, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            return ProviderError(self.name, ErrorKind.AUTH, str(e))
        if isinstance(e, anthropic.APIStatusError):
            status = getattr(e, "status_code", None) or e.response.status_code
            return ProviderError(self.name, classify_status(status), str(e), None, status)
        if isinstance(e, anthropic.APIConnectionError):  # includes APITimeoutError
            return ProviderError(self.name, ErrorKind.TRANSIENT, str(e))
        return ProviderError(self.name, ErrorKind.TRANSIENT, repr(e))

    async def stream(self, request: LLMRequest) -> AsyncIterator[str | Usage]:
        produced = False
        try:
            async with self._client.beta.messages.stream(**self._kwargs(request)) as s:
                async for text in s.text_stream:
                    if text:
                        produced = True
                        yield text
                final = await s.get_final_message()
        except anthropic.AnthropicError as e:
            raise self._map_error(e) from e
        if final.stop_reason == "refusal" and not produced:
            raise ProviderError(self.name, ErrorKind.REFUSAL, "declined by safety classifier")
        yield Usage(final.usage.input_tokens, final.usage.output_tokens)

    async def complete(self, request: LLMRequest) -> LLMResult:
        started = time.perf_counter()
        parts: list[str] = []
        usage = Usage()
        async for item in self.stream(request):
            if isinstance(item, Usage):
                usage = item
            else:
                parts.append(item)
        text = "".join(parts).strip()
        if not text:
            raise ProviderError(self.name, ErrorKind.EMPTY, "empty content")
        return LLMResult(text, self.name, self.model, usage, (time.perf_counter() - started) * 1000)
