"""Smoke-test every configured LLM provider with one tiny request each.

    python -m src.llm.check            # all providers
    python -m src.llm.check --stream   # also exercise streaming

Each provider is called directly (not via failover), so a broken key or retired model shows up as
that provider's own error instead of being hidden by the router.
"""

import argparse
import asyncio
import time

from src.core.config import settings
from src.llm.base import ChatMessage, LLMRequest, ProviderError, Usage
from src.llm.registry import build_adapter, resolve_specs


async def _check(spec, stream: bool) -> str:
    adapter = build_adapter(spec, settings)
    request = LLMRequest(
        [ChatMessage("user", "Reply with exactly: ok")], max_tokens=20, temperature=0.0
    )
    started = time.perf_counter()
    try:
        if stream:
            parts = [i async for i in adapter.stream(request) if not isinstance(i, Usage)]
            text = "".join(parts)
        else:
            text = (await adapter.complete(request)).text
        ms = (time.perf_counter() - started) * 1000
        return f"OK    {spec.name:<40} {ms:6.0f} ms  {text.strip()[:40]!r}"
    except ProviderError as e:
        return f"FAIL  {spec.name:<40} {e.kind.value}: {e.message[:120]}"
    finally:
        await adapter.aclose()


async def main(stream: bool) -> None:
    specs = resolve_specs(settings)
    print(f"{len(specs)} provider(s) configured")
    for spec in specs:  # sequential: stays inside every free tier's rate limit
        print(await _check(spec, stream))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Check each configured LLM provider")
    cli.add_argument("--stream", action="store_true")
    asyncio.run(main(cli.parse_args().stream))
