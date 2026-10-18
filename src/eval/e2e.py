"""End-to-end scenarios against a RUNNING server, real LLMs, real Postgres and Qdrant.

    python -m src.eval.e2e [--base http://127.0.0.1:8001] [--only auth,tutor] [--pace 6]

What the offline suite cannot show: that a real model follows the output contracts (one question,
the META trailer, [n] citations), that the real database stores what the fakes accept, and that
the agents behave on edge cases. Every run registers throwaway users and erases them at the end
(through DELETE /me, so the GDPR path is exercised too). The corpus is whatever is indexed, so
scenarios are phrased from the papers actually present and discovered at start-up.

Results are PASS / FAIL (a contract was broken) / NOTE (behaviour worth knowing, not a defect).
Exit status is non-zero if anything FAILed.
"""

import argparse
import asyncio
import json
import re
import secrets
import sys
import time
from dataclasses import dataclass, field

import httpx

CITE = re.compile(r"\[(\d+)(?:\s*[,\-–]\s*\d+)*\]")


@dataclass
class Turn:
    status: int = 0
    headers: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    intent: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    provider: dict = field(default_factory=dict)
    done: dict = field(default_factory=dict)
    error: dict = field(default_factory=dict)
    text: str = ""
    events: list = field(default_factory=list)
    seconds: float = 0.0
    first_token: float | None = None

    @property
    def conversation_id(self) -> str | None:
        return self.meta.get("conversation_id")

    @property
    def answer(self) -> str:
        return self.done.get("answer") or self.text


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.section = ""

    def start(self, name: str) -> None:
        self.section = name
        print(f"\n== {name}", flush=True)

    def _add(self, status: str, name: str, detail: str) -> None:
        self.rows.append((self.section, status, name, detail))
        print(f"  {status:4} {name}" + (f"  [{detail}]" if detail else ""), flush=True)

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self._add("PASS" if ok else "FAIL", name, "" if ok else detail)
        return ok

    def note(self, name: str, detail: str = "") -> None:
        self._add("NOTE", name, detail)

    @property
    def failed(self) -> list:
        return [r for r in self.rows if r[1] == "FAIL"]


class Api:
    def __init__(self, base: str, pace: float):
        self.base = base.rstrip("/") + "/api/v1"
        self.http = httpx.AsyncClient(timeout=120)
        self.pace = pace
        self.created: list[tuple[str, str]] = []

    async def aclose(self) -> None:
        await self.http.aclose()

    async def register(self, tag: str = "u") -> tuple[dict, str]:
        email = f"e2e-{tag}-{secrets.token_hex(3)}@example.test"
        password = secrets.token_urlsafe(14)
        r = await self.http.post(
            f"{self.base}/auth/register", json={"email": email, "password": password}
        )
        r.raise_for_status()
        self.created.append((email, password))
        return {"Authorization": f"Bearer {r.json()['access_token']}"}, email

    async def get(self, path: str, headers: dict | None = None, **kw) -> httpx.Response:
        return await self.http.get(f"{self.base}{path}", headers=headers, **kw)

    async def post(self, path: str, headers: dict | None, body: dict) -> httpx.Response:
        return await self.http.post(f"{self.base}{path}", headers=headers, json=body)

    async def turn(self, headers: dict | None, message: str, **extra) -> Turn:
        """One streamed chat turn, parsed into its events. Paced to respect free-tier limits."""
        await asyncio.sleep(self.pace)
        t = Turn()
        started = time.perf_counter()
        body = {"message": message, **{k: v for k, v in extra.items() if v is not None}}
        async with self.http.stream(
            "POST", f"{self.base}/chat/stream", headers=headers, json=body
        ) as r:
            t.status, t.headers = r.status_code, dict(r.headers)
            if r.status_code != 200:
                await r.aread()
                try:
                    t.error = r.json()
                except ValueError:
                    t.error = {"raw": r.text[:200]}
                t.seconds = time.perf_counter() - started
                return t
            event = None
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event:
                    data = json.loads(line[5:])
                    if event == "token":
                        if t.first_token is None:
                            t.first_token = time.perf_counter() - started
                        t.text += data.get("text") or data.get("delta") or ""
                        continue
                    t.events.append(event)
                    if event == "sources":
                        t.sources = data.get("sources", [])
                    elif event in ("meta", "intent", "provider", "done", "error"):
                        setattr(t, event, data)
        t.seconds = time.perf_counter() - started
        return t


def cited(text: str) -> set[int]:
    nums: set[int] = set()
    for m in re.finditer(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]", text):
        for part in re.split(r"\s*,\s*", m.group(1)):
            a = re.split(r"\s*[\-–]\s*", part)
            nums.update(range(int(a[0]), int(a[-1]) + 1))
    return nums


async def _cleanup(api: Api) -> int:
    """Erase every throwaway user through the real GDPR endpoint."""
    erased = 0
    for email, password in api.created:
        login = await api.http.post(
            f"{api.base}/auth/login", json={"email": email, "password": password}
        )
        if login.status_code != 200:
            continue
        h = {"Authorization": f"Bearer {login.json()['access_token']}"}
        r = await api.http.request(
            "DELETE", f"{api.base}/me", headers=h, json={"password": password}
        )
        erased += r.status_code == 204
    return erased


async def run(base: str, only: list[str] | None, pace: float) -> int:
    from src.eval.e2e_scenarios import SCENARIOS

    api, rep = Api(base, pace), Report()
    try:
        h, _ = await api.register("setup")
        corpus = (await api.get("/papers/map", h)).json()["nodes"]
        corpus = [{"arxiv_id": n["id"], "title": n["title"]} for n in corpus]
        print(f"server {base}: {len(corpus)} papers indexed")
        for name, scenario in SCENARIOS.items():
            if only and name not in only:
                continue
            api.pace = pace
            try:
                await scenario(api, rep, corpus)
            except Exception as e:  # noqa: BLE001 - a crashed scenario is a failure, not a stop
                rep.check(False, f"scenario {name} ran to the end", f"{type(e).__name__}: {e}")
    finally:
        erased = await _cleanup(api)
        await api.aclose()
    counts = {s: sum(1 for r in rep.rows if r[1] == s) for s in ("PASS", "FAIL", "NOTE")}
    print(
        f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['NOTE']} notes; "
        f"{erased} throwaway users erased"
    )
    for section, _, name, detail in rep.failed:
        print(f"  FAIL [{section}] {name}: {detail}")
    return 1 if rep.failed else 0


def main() -> None:
    cli = argparse.ArgumentParser(description="End-to-end scenarios against a running server")
    cli.add_argument("--base", default="http://127.0.0.1:8001")
    cli.add_argument("--only", help="comma-separated scenario names")
    cli.add_argument("--pace", type=float, default=4.0, help="seconds between LLM turns")
    args = cli.parse_args()
    if reconfigure := getattr(sys.stdout, "reconfigure", None):  # Windows consoles are cp1252
        reconfigure(encoding="utf-8", errors="replace")
    only = args.only.split(",") if args.only else None
    sys.exit(asyncio.run(run(args.base, only, args.pace)))


if __name__ == "__main__":
    main()
