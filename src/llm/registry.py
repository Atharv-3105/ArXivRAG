"""Builds the LLM router from settings.

Default behaviour: every provider whose API key is set becomes a slot, with free-tier limits.
`LLM_PROVIDERS_JSON` replaces the defaults entirely, e.g.

    [{"name": "groq", "kind": "openai_compat", "base_url": "https://api.groq.com/openai/v1",
      "model": "openai/gpt-oss-120b", "api_key_env": "GROQ_API_KEY", "rpm": 30, "tpm": 8000}]

Keys are referenced by env-var name (`api_key_env`), never inlined in the JSON.

The rpm/tpm numbers below are conservative free-tier defaults; providers change them, so they are
config (override per provider via LLM_PROVIDERS_JSON), not truth.
"""

import json
import os
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from src.core.config import Settings, settings
from src.llm.anthropic_adapter import AnthropicAdapter
from src.llm.base import LLMAdapter
from src.llm.mock import MockAdapter
from src.llm.openai_compat import OpenAICompatAdapter
from src.llm.router import LLMRouter, ProviderLimits

logger = structlog.get_logger(__name__)


class ProviderSpec(BaseModel):
    name: str
    kind: Literal["openai_compat", "anthropic", "mock"] = "openai_compat"
    model: str = "mock-extractive-1"
    base_url: str | None = None
    api_key: str | None = Field(default=None, exclude=True)
    api_key_env: str | None = None
    rpm: int | None = None
    tpm: int | None = None
    rate_group: str | None = None
    timeout: float | None = None
    stream_usage: bool = False
    supports_temperature: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)

    def resolved_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.getenv(self.api_key_env) if self.api_key_env else None


def _secret(value) -> str | None:
    return value.get_secret_value() if value is not None else None


def default_specs(cfg: Settings) -> list[ProviderSpec]:
    specs: list[ProviderSpec] = []
    if key := _secret(cfg.GROQ_API_KEY):
        specs.append(
            ProviderSpec(
                name="groq",
                base_url="https://api.groq.com/openai/v1",
                model=cfg.GROQ_MODEL,
                api_key=key,
                rpm=30,
                tpm=8000,
                stream_usage=True,
            )
        )
    if key := _secret(cfg.GEMINI_API_KEY):
        specs.append(
            ProviderSpec(
                name="gemini",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                model=cfg.GEMINI_MODEL,
                api_key=key,
                rpm=10,
                tpm=250_000,
            )
        )
    if key := _secret(cfg.CEREBRAS_API_KEY):
        specs.append(
            ProviderSpec(
                name="cerebras",
                base_url="https://api.cerebras.ai/v1",
                model=cfg.CEREBRAS_MODEL,
                api_key=key,
                rpm=30,
                tpm=60_000,
            )
        )
    if key := _secret(cfg.OPENROUTER_API_KEY):
        # One slot per free model so a deprecated/overloaded model fails over to the next, but ONE
        # shared rate group: OpenRouter's free limit is per account, not per model.
        for model in cfg.OPENROUTER_MODELS:
            specs.append(
                ProviderSpec(
                    name=f"openrouter:{model}",
                    base_url="https://openrouter.ai/api/v1",
                    model=model,
                    api_key=key,
                    rpm=20,
                    rate_group="openrouter",
                    stream_usage=True,
                    extra_headers={"HTTP-Referer": cfg.PUBLIC_BASE_URL, "X-Title": cfg.APP_NAME},
                )
            )
    if cfg.OLLAMA_BASE_URL:
        specs.append(
            ProviderSpec(
                name="ollama",
                base_url=cfg.OLLAMA_BASE_URL,
                model=cfg.OLLAMA_MODEL,
                timeout=180.0,
            )
        )
    if key := _secret(cfg.ANTHROPIC_API_KEY):
        specs.append(
            ProviderSpec(
                name="anthropic",
                kind="anthropic",
                model=cfg.ANTHROPIC_MODEL,
                api_key=key,
                rpm=50,
            )
        )
    return specs


def build_adapter(spec: ProviderSpec, cfg: Settings) -> LLMAdapter:
    timeout = spec.timeout or cfg.LLM_TIMEOUT_SECONDS
    if spec.kind == "mock":
        return MockAdapter(spec.name, spec.model)
    key = spec.resolved_key()
    if spec.kind == "anthropic":
        if not key:
            raise ValueError(f"provider {spec.name}: no API key")
        return AnthropicAdapter(
            spec.name,
            spec.model,
            key,
            timeout=timeout,
            server_fallbacks=cfg.ANTHROPIC_SERVER_FALLBACKS,
            default_retry_after=cfg.LLM_DEFAULT_RETRY_AFTER_SECONDS,
        )
    if not spec.base_url:
        raise ValueError(f"provider {spec.name}: base_url is required")
    return OpenAICompatAdapter(
        spec.name,
        spec.base_url,
        spec.model,
        api_key=key,
        timeout=timeout,
        extra_headers=spec.extra_headers,
        stream_usage=spec.stream_usage,
        supports_temperature=spec.supports_temperature,
        default_retry_after=cfg.LLM_DEFAULT_RETRY_AFTER_SECONDS,
    )


def resolve_specs(cfg: Settings) -> list[ProviderSpec]:
    specs = (
        [ProviderSpec(**s) for s in json.loads(cfg.LLM_PROVIDERS_JSON)]
        if cfg.LLM_PROVIDERS_JSON
        else default_specs(cfg)
    )
    real = [s for s in specs if s.kind != "mock"]
    use_mock = cfg.LLM_ENABLE_MOCK if cfg.LLM_ENABLE_MOCK is not None else not real
    if use_mock and not any(s.kind == "mock" for s in specs):
        specs.append(ProviderSpec(name="mock", kind="mock"))
    if not specs:
        raise ValueError("No LLM provider configured and the mock is disabled (LLM_ENABLE_MOCK)")
    return specs


def build_router(cfg: Settings = settings, observer=None) -> LLMRouter:
    specs = resolve_specs(cfg)
    slots = [
        (build_adapter(s, cfg), ProviderLimits(rpm=s.rpm, tpm=s.tpm, group=s.rate_group))
        for s in specs
    ]
    logger.info("LLM router ready", providers=[s.name for s in specs])
    return LLMRouter(
        slots,
        cfg.LLM_TASK_ORDERS,
        failure_threshold=cfg.LLM_BREAKER_FAILURE_THRESHOLD,
        recovery_seconds=cfg.LLM_BREAKER_RECOVERY_SECONDS,
        max_recovery_seconds=cfg.LLM_BREAKER_MAX_RECOVERY_SECONDS,
        call_timeout=cfg.LLM_TIMEOUT_SECONDS,
        first_token_timeout=cfg.LLM_FIRST_TOKEN_TIMEOUT_SECONDS,
        idle_timeout=cfg.LLM_STREAM_IDLE_TIMEOUT_SECONDS,
        observer=observer,
    )
