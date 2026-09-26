"""Adapter for any OpenAI-compatible chat-completions API.

One adapter covers Groq, Google Gemini (its /v1beta/openai endpoint), Cerebras, OpenRouter, Ollama
and OpenAI itself, so the free-tier providers need no vendor SDKs.
"""

import json
import time
from collections.abc import AsyncIterator

import httpx

from src.llm.base import (
    ErrorKind,
    LLMRequest,
    LLMResult,
    ProviderError,
    Usage,
    classify_status,
    parse_retry_after,
)


class OpenAICompatAdapter:
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        stream_usage: bool = True,
        supports_temperature: bool = True,
        default_retry_after: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.name = name
        self.model = model
        self.stream_usage = stream_usage
        self.supports_temperature = supports_temperature
        self.default_retry_after = default_retry_after
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # One pooled client per provider: TLS connections are reused across requests.
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, request: LLMRequest, stream: bool) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if self.supports_temperature:
            body["temperature"] = request.temperature
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream and self.stream_usage:
            body["stream_options"] = {"include_usage": True}
        return body

    def _error(self, resp: httpx.Response) -> ProviderError:
        kind = classify_status(resp.status_code)
        retry = (
            parse_retry_after(dict(resp.headers), self.default_retry_after)
            if kind is ErrorKind.RATE_LIMITED
            else None
        )
        return ProviderError(self.name, kind, _short(resp.text), retry, resp.status_code)

    def _body_error(self, data: dict) -> ProviderError:
        """Some gateways (OpenRouter) answer HTTP 200 with {"error": {...}}."""
        err = data.get("error") or {}
        code = err.get("code") if isinstance(err, dict) else None
        kind = classify_status(code) if isinstance(code, int) else ErrorKind.TRANSIENT
        retry = self.default_retry_after if kind is ErrorKind.RATE_LIMITED else None
        return ProviderError(self.name, kind, _short(json.dumps(err)), retry, code)

    async def complete(self, request: LLMRequest) -> LLMResult:
        started = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=self._payload(request, False))
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, ErrorKind.TRANSIENT, f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise ProviderError(self.name, ErrorKind.TRANSIENT, f"network: {e}") from e
        if resp.status_code >= 400:
            raise self._error(resp)
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(self.name, ErrorKind.TRANSIENT, "non-JSON response") from e
        if "choices" not in data:
            raise self._body_error(data)

        choice = data["choices"][0] if data["choices"] else {}
        text = ((choice.get("message") or {}).get("content") or "").strip()
        if choice.get("finish_reason") == "content_filter":
            raise ProviderError(self.name, ErrorKind.REFUSAL, "content filtered")
        if not text:
            raise ProviderError(self.name, ErrorKind.EMPTY, "empty content")
        u = data.get("usage") or {}
        return LLMResult(
            text=text,
            provider=self.name,
            model=data.get("model") or self.model,
            usage=Usage(u.get("prompt_tokens"), u.get("completion_tokens")),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str | Usage]:
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=self._payload(request, True)
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise self._error(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except ValueError:
                        continue  # keep-alive or partial noise
                    if "error" in data and "choices" not in data:
                        raise self._body_error(data)
                    for choice in data.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield delta
                        if choice.get("finish_reason") == "content_filter":
                            raise ProviderError(self.name, ErrorKind.REFUSAL, "content filtered")
                    if data.get("usage"):
                        u = data["usage"]
                        yield Usage(u.get("prompt_tokens"), u.get("completion_tokens"))
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, ErrorKind.TRANSIENT, f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise ProviderError(self.name, ErrorKind.TRANSIENT, f"network: {e}") from e


def _short(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]
