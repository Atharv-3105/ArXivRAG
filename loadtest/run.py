"""HTTP load test against a running deployment (no extra dependencies: asyncio + httpx).

    python loadtest/run.py --base-url http://localhost:8000 --users 10 --turns 5

Each virtual user registers, then sends `turns` streamed chat messages. Reports p50/p95/p99 of
time to first token and full turn, the error rate and the status codes seen. Point it at staging,
not production: every turn spends real LLM quota. With FREE_DAILY_MESSAGES and
RATE_LIMIT_PER_MINUTE in force, expect 429s past the limits - that is the limiter working.
"""

import argparse
import asyncio
import json
import math
import time
import uuid
from collections import Counter

import httpx

QUERIES = [
    "Explain how LoRA works",
    "Compare LoRA and QLoRA",
    "Critique the baselines of the QLoRA paper",
    "What should I read next?",
    "How does attention work in transformers?",
]


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q / 100 * len(ordered))) - 1]


async def user(client: httpx.AsyncClient, turns: int, stats: dict) -> None:
    email = f"load-{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": "load-test!"})
    if r.status_code != 201:
        stats["codes"][f"register:{r.status_code}"] += 1
        return
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    for i in range(turns):
        started = time.perf_counter()
        first = None
        try:
            async with client.stream(
                "POST",
                "/api/v1/chat/stream",
                json={"message": QUERIES[i % len(QUERIES)]},
                headers=headers,
            ) as resp:
                stats["codes"][str(resp.status_code)] += 1
                async for line in resp.aiter_lines():
                    if first is None and line == "event: token":
                        first = time.perf_counter() - started
                    if line == "event: error":
                        stats["stream_errors"] += 1
        except httpx.HTTPError as e:
            stats["codes"][type(e).__name__] += 1
            continue
        stats["turn"].append(time.perf_counter() - started)
        if first is not None:
            stats["ttft"].append(first)


async def main(base_url: str, users: int, turns: int) -> dict:
    stats: dict = {"turn": [], "ttft": [], "codes": Counter(), "stream_errors": 0}
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        wall = time.perf_counter()
        await asyncio.gather(*(user(client, turns, stats) for _ in range(users)))
        wall = time.perf_counter() - wall
    total = sum(stats["codes"].values()) or 1
    errors = sum(n for code, n in stats["codes"].items() if not code.startswith("2"))
    return {
        "users": users,
        "turns_per_user": turns,
        "wall_s": round(wall, 1),
        "ttft_p50_s": round(pct(stats["ttft"], 50), 3),
        "ttft_p95_s": round(pct(stats["ttft"], 95), 3),
        "turn_p95_s": round(pct(stats["turn"], 95), 3),
        "turn_p99_s": round(pct(stats["turn"], 99), 3),
        "error_rate": round(errors / total, 4),
        "stream_errors": stats["stream_errors"],
        "status_codes": dict(stats["codes"]),
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base-url", default="http://localhost:8000")
    cli.add_argument("--users", type=int, default=10)
    cli.add_argument("--turns", type=int, default=5)
    args = cli.parse_args()
    print(json.dumps(asyncio.run(main(args.base_url, args.users, args.turns)), indent=2))
