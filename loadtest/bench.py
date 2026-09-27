"""In-process latency benchmark: the application's own overhead, with the LLM and stores faked.

    python -m loadtest.bench [--turns 200] [--concurrency 10]

Real latency = provider time + our overhead. Provider time is out of our hands (and is measured by
the live suite and the arxivmind_llm_latency_seconds metric); overhead - auth, routing, graph,
guardrail, serialization, SSE - is ours, and this measures it without any network. If overhead
alone threatens the 2 s p95 budget, no provider choice can save it.
"""

import argparse
import asyncio
import os
import time

os.environ.setdefault("SECRET_KEY", "bench-secret-key-not-for-production-use")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://bench:bench@localhost:5432/bench")
os.environ.setdefault("BCRYPT_ROUNDS", "4")


async def run(turns: int, concurrency: int) -> dict:
    import httpx

    from src.api.app import create_app
    from src.core.config import settings
    from src.eval import scoring
    from tests.chat_fakes import hit, make_services  # the same offline fakes the tests use

    settings.RATE_LIMIT_PER_MINUTE = 10**9
    settings.FREE_DAILY_MESSAGES = 10**9
    hits = [hit(f"2106.0968{i}v1", i) for i in range(8)]
    services = make_services(hits)
    app = create_app(services=services)
    latencies: list[float] = []
    ttft: list[float] = []
    queries = ["Explain how LoRA works", "Compare LoRA and QLoRA", "Critique LoRA's baselines"]

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bench") as client:
            r = await client.post(
                "/api/v1/auth/register",
                json={"email": "bench@example.com", "password": "benchmark!"},
            )
            client.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
            sem = asyncio.Semaphore(concurrency)

            async def one(i: int) -> None:
                async with sem:
                    started = time.perf_counter()
                    first = None
                    async with client.stream(
                        "POST", "/api/v1/chat/stream", json={"message": queries[i % len(queries)]}
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if first is None and line == "event: token":
                                first = time.perf_counter() - started
                    latencies.append(time.perf_counter() - started)
                    ttft.append(first or 0.0)

            wall = time.perf_counter()
            await asyncio.gather(*(one(i) for i in range(turns)))
            wall = time.perf_counter() - wall
    ms = lambda v: round(v * 1000, 1)  # noqa: E731
    return {
        "turns": turns,
        "concurrency": concurrency,
        "throughput_turns_per_s": round(turns / wall, 1),
        "overhead_p50_ms": ms(scoring.percentile(latencies, 50)),
        "overhead_p95_ms": ms(scoring.percentile(latencies, 95)),
        "overhead_p99_ms": ms(scoring.percentile(latencies, 99)),
        "first_token_p95_ms": ms(scoring.percentile(ttft, 95)),
    }


def main() -> None:
    cli = argparse.ArgumentParser(description="Application overhead benchmark (no network)")
    cli.add_argument("--turns", type=int, default=200)
    cli.add_argument("--concurrency", type=int, default=10)
    args = cli.parse_args()
    import json

    print(json.dumps(asyncio.run(run(args.turns, args.concurrency)), indent=2))


if __name__ == "__main__":
    main()
