"""Provider adapters against mocked transports, plus the registry that wires them up."""

import json
from types import SimpleNamespace

import httpx
import pytest

from src.agents.trailer import split_complete
from src.core.config import Settings
from src.llm.anthropic_adapter import AnthropicAdapter
from src.llm.base import ChatMessage, ErrorKind, LLMRequest, ProviderError, Usage, parse_retry_after
from src.llm.mock import MockAdapter
from src.llm.openai_compat import OpenAICompatAdapter
from src.llm.registry import build_router, resolve_specs


def req(**kw):
    return LLMRequest(
        [ChatMessage("system", "be brief"), ChatMessage("user", "hello")],
        max_tokens=50,
        **kw,
    )


def adapter(handler, **kw):
    return OpenAICompatAdapter(
        "p", "https://x.test/v1", "m", api_key="k", transport=httpx.MockTransport(handler), **kw
    )


def ok_body(text="hi there", usage=True):
    body = {
        "model": "m-served",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 7, "completion_tokens": 3}
    return body


# ---------------------------------------------------------------- OpenAI-compatible


async def test_complete_sends_openai_shape_and_parses_usage():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    result = await adapter(handler).complete(req(json_mode=True))
    assert seen["url"] == "https://x.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["stream"] is False
    assert (result.text, result.model, result.usage.total) == ("hi there", "m-served", 10)


async def test_temperature_is_omitted_for_providers_that_reject_it():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    await adapter(handler, supports_temperature=False).complete(req())
    assert "temperature" not in seen


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (429, ErrorKind.RATE_LIMITED),
        (401, ErrorKind.AUTH),
        (403, ErrorKind.AUTH),
        (400, ErrorKind.BAD_REQUEST),
        (404, ErrorKind.BAD_REQUEST),
        (500, ErrorKind.TRANSIENT),
        (503, ErrorKind.TRANSIENT),
        (408, ErrorKind.TRANSIENT),
    ],
)
async def test_http_errors_are_classified_by_status_code(status, kind):
    a = adapter(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(ProviderError) as exc:
        await a.complete(req())
    assert exc.value.kind is kind and exc.value.status_code == status


async def test_rate_limit_reads_retry_after_header_not_free_text():
    """The body mentions 3599 tokens; only the header decides the cooldown."""
    a = adapter(
        lambda r: httpx.Response(
            429, headers={"retry-after": "17"}, text="limit 3599 tokens, try later"
        )
    )
    with pytest.raises(ProviderError) as exc:
        await a.complete(req())
    assert exc.value.retry_after == 17


def test_retry_after_parsing_supports_provider_duration_headers():
    assert parse_retry_after({"x-ratelimit-reset-requests": "2m59.5s"}, 60) == pytest.approx(179.5)
    assert parse_retry_after({"x-ratelimit-reset-tokens": "450ms"}, 60) == pytest.approx(0.45)
    assert parse_retry_after({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 60) == 60
    assert parse_retry_after({}, 42) == 42


async def test_http_200_with_an_error_body_is_an_error():
    """OpenRouter-style: status 200 but {"error": {...}} and no choices."""
    body = {"error": {"code": 429, "message": "free model busy"}}
    with pytest.raises(ProviderError) as exc:
        await adapter(lambda r: httpx.Response(200, json=body)).complete(req())
    assert exc.value.kind is ErrorKind.RATE_LIMITED


async def test_empty_content_is_an_error():
    with pytest.raises(ProviderError) as exc:
        await adapter(lambda r: httpx.Response(200, json=ok_body(text="  "))).complete(req())
    assert exc.value.kind is ErrorKind.EMPTY


async def test_content_filter_is_a_refusal():
    body = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
    with pytest.raises(ProviderError) as exc:
        await adapter(lambda r: httpx.Response(200, json=body)).complete(req())
    assert exc.value.kind is ErrorKind.REFUSAL


async def test_network_failure_is_transient():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError) as exc:
        await adapter(handler).complete(req())
    assert exc.value.kind is ErrorKind.TRANSIENT


def sse(*chunks):
    return "".join(f"data: {c}\n\n" for c in chunks)


async def test_stream_parses_sse_deltas_and_final_usage():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        body = sse(
            json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
            json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
            ": keep-alive",
            json.dumps({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
            json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
            "[DONE]",
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    items = [i async for i in adapter(handler, stream_usage=True).stream(req())]
    assert "".join(i for i in items if isinstance(i, str)) == "Hello"
    assert [i.total for i in items if isinstance(i, Usage)] == [7]
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}


async def test_stream_error_status_is_classified():
    a = adapter(lambda r: httpx.Response(429, headers={"retry-after": "5"}, text="slow down"))
    with pytest.raises(ProviderError) as exc:
        _ = [i async for i in a.stream(req())]
    assert exc.value.kind is ErrorKind.RATE_LIMITED and exc.value.retry_after == 5


# ---------------------------------------------------------------------- Anthropic


class FakeStream:
    def __init__(self, texts, stop_reason="end_turn"):
        self.texts = texts
        self.final = SimpleNamespace(
            stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=9, output_tokens=4)
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for t in self.texts:
                yield t

        return gen()

    async def get_final_message(self):
        return self.final


class FakeAnthropicClient:
    def __init__(self, stream):
        self.kwargs = None
        self._stream = stream
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._make))

    def _make(self, **kwargs):
        self.kwargs = kwargs
        return self._stream

    async def close(self):
        pass


async def test_anthropic_request_shape_and_usage():
    client = FakeAnthropicClient(FakeStream(["Hel", "lo"]))
    a = AnthropicAdapter("anthropic", "claude-opus-5", "k", client=client)
    result = await a.complete(req())
    assert result.text == "Hello" and result.usage.total == 13
    kw = client.kwargs
    assert kw["system"] == "be brief"
    assert kw["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in kw  # current Claude models reject sampling params
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]


async def test_anthropic_fallbacks_are_only_sent_for_supporting_models():
    client = FakeAnthropicClient(FakeStream(["ok"]))
    await AnthropicAdapter("anthropic", "claude-haiku-4-5", "k", client=client).complete(req())
    assert "fallbacks" not in client.kwargs and "betas" not in client.kwargs


async def test_anthropic_refusal_with_no_text_is_a_refusal():
    client = FakeAnthropicClient(FakeStream([], stop_reason="refusal"))
    with pytest.raises(ProviderError) as exc:
        await AnthropicAdapter("anthropic", "claude-opus-5", "k", client=client).complete(req())
    assert exc.value.kind is ErrorKind.REFUSAL


# --------------------------------------------------------------------------- mock


async def test_mock_quotes_and_cites_numbered_sources():
    r = LLMRequest(
        [ChatMessage("user", "Sources:\n[1] Transformers use attention. More.\n[2] RNNs recur.")]
    )
    text = (await MockAdapter().complete(r)).text
    assert "Transformers use attention. [1]" in text and "[2]" in text


async def test_mock_stream_matches_complete():
    r = LLMRequest([ChatMessage("user", "[1] A fact.")], task="tutor", tags={"agent": "tutor"})
    m = MockAdapter()
    streamed = "".join([i async for i in m.stream(r) if isinstance(i, str)])
    assert streamed == (await m.complete(r)).text
    visible, meta = split_complete(streamed)
    assert visible.endswith("?") and meta["question"] == visible.splitlines()[-1]


# ----------------------------------------------------------------------- registry


def cfg(**kw):
    return Settings(_env_file=None, SECRET_KEY="x", **kw)


def test_no_keys_means_mock_only():
    assert [s.name for s in resolve_specs(cfg())] == ["mock"]


def test_each_configured_key_enables_its_provider_and_mock_is_off():
    names = [
        s.name
        for s in resolve_specs(
            cfg(
                GROQ_API_KEY="g",
                GEMINI_API_KEY="m",
                OPENROUTER_API_KEY="o",
                OPENROUTER_MODELS=["a:free", "b:free"],
            )
        )
    ]
    assert names == ["groq", "gemini", "openrouter:a:free", "openrouter:b:free"]


def test_openrouter_models_share_one_rate_group():
    specs = resolve_specs(cfg(OPENROUTER_API_KEY="o", OPENROUTER_MODELS=["a:free", "b:free"]))
    assert {s.rate_group for s in specs} == {"openrouter"}


def test_mock_can_be_forced_on_as_a_last_resort():
    specs = resolve_specs(cfg(GROQ_API_KEY="g", LLM_ENABLE_MOCK=True))
    assert [s.name for s in specs] == ["groq", "mock"]


def test_json_override_references_keys_by_env_var(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    raw = json.dumps(
        [
            {
                "name": "custom",
                "base_url": "https://c/v1",
                "model": "x",
                "api_key_env": "MY_KEY",
                "rpm": 5,
            }
        ]
    )
    [spec] = [s for s in resolve_specs(cfg(LLM_PROVIDERS_JSON=raw)) if s.name == "custom"]
    assert spec.resolved_key() == "secret" and spec.rpm == 5


async def test_build_router_with_keys_orders_by_task():
    router = build_router(cfg(GROQ_API_KEY="g", GEMINI_API_KEY="m"))
    assert [s.name for s in router.order_for("router")] == ["groq", "gemini"]
    assert [s.name for s in router.order_for("tutor")] == ["gemini", "groq"]
    assert not router.only_mock
    await router.aclose()
