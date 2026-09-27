"""Prometheus metrics (ADR 0012). Scraped at GET /metrics.

Label sets are small and bounded on purpose (route *templates*, agent names, provider slots,
issue codes): user ids, queries or paper ids as labels would explode the series count, which is
the classic way to take down a Prometheus.

Runtime state that lives in objects (the memory queue, circuit breakers) is exported by a
collector that reads it at scrape time, so the hot path never touches a gauge for it.
"""

import time

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60)

HTTP_REQUESTS = Counter(
    "arxivmind_http_requests_total",
    "HTTP requests",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "arxivmind_http_request_duration_seconds",
    "Full request duration (for SSE: until the stream ends)",
    ["route"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
CHAT_TURNS = Counter(
    "arxivmind_chat_turns_total",
    "Chat turns by specialist and outcome",
    ["agent", "action", "provider", "outcome"],
    registry=REGISTRY,
)
FIRST_TOKEN = Histogram(
    "arxivmind_chat_first_token_seconds",
    "Time from request to the first streamed answer token (the latency users feel)",
    ["agent"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
LLM_CALLS = Counter(
    "arxivmind_llm_calls_total",
    "LLM provider calls by outcome",
    ["provider", "outcome"],
    registry=REGISTRY,
)
LLM_LATENCY = Histogram(
    "arxivmind_llm_latency_seconds",
    "Successful LLM call latency",
    ["provider"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
GUARDRAIL = Counter(
    "arxivmind_guardrail_decisions_total",
    "Guardrail decisions; issue='none' when approved clean",
    ["approved", "issue"],
    registry=REGISTRY,
)
INPUT_GATE = Counter(
    "arxivmind_input_gate_total", "Input gate outcomes", ["outcome"], registry=REGISTRY
)
CHECK_FAILURES = Counter(
    "arxivmind_output_check_failures_total",
    "Specialist output checks that failed",
    ["agent", "check"],
    registry=REGISTRY,
)
GRAPH_RETRIEVAL = Counter(
    "arxivmind_graph_retrieval_total",
    "Graph-enhanced retrieval status",
    ["status"],
    registry=REGISTRY,
)
RATE_LIMITED = Counter(
    "arxivmind_rate_limited_total", "Requests rejected by limits", ["kind"], registry=REGISTRY
)


def observe_llm(provider: str, outcome: str, latency_ms: float | None) -> None:
    """LLMRouter observer hook (see router._observe)."""
    LLM_CALLS.labels(provider, outcome).inc()
    if latency_ms is not None:
        LLM_LATENCY.labels(provider).observe(latency_ms / 1000)


class RuntimeCollector:
    """Reads live objects at scrape time. `services` is set by the app lifespan."""

    services = None

    def collect(self):
        svc = RuntimeCollector.services
        if svc is None:
            return
        memory = getattr(svc, "memory", None)
        if memory is not None:
            depth = GaugeMetricFamily("arxivmind_memory_queue_depth", "Queued memory events")
            depth.add_metric([], memory.queue.qsize())
            yield depth
            stats = GaugeMetricFamily(
                "arxivmind_memory_events", "MemoryManager counters since start", labels=["kind"]
            )
            for kind, value in vars(memory.stats).items():
                stats.add_metric([kind], value)
            yield stats
        llm = getattr(svc, "llm", None)
        if llm is not None and hasattr(llm, "status"):
            breaker = GaugeMetricFamily(
                "arxivmind_llm_breaker_open",
                "1 if the provider's circuit breaker is open or half-open",
                labels=["provider"],
            )
            for name, row in llm.status().items():
                breaker.add_metric([name], 0 if row.get("breaker") == "closed" else 1)
            yield breaker


REGISTRY.register(RuntimeCollector())


def route_template(scope: Scope) -> str:
    """'/api/v1/conversations/{conversation_id}', never the concrete path (cardinality).

    Included routers are mounted, so scope["route"].path is relative to the router's prefix
    ('/conversations/{conversation_id}'). The prefix is whatever precedes the concrete match.
    """
    route = scope.get("route")
    path_format = getattr(route, "path_format", None)
    if route is None or path_format is None:
        return "unmatched"
    try:
        concrete = path_format.format(**scope.get("path_params", {}))
    except (KeyError, IndexError, ValueError):
        return route.path
    path = scope.get("path", "")
    prefix = path[: len(path) - len(concrete)] if path.endswith(concrete) else ""
    return prefix + route.path


class MetricsMiddleware:
    """Pure ASGI (SSE-safe, like TraceIDMiddleware). Labels by route template, not raw path."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = {"code": 500}

        async def capture(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            template = route_template(scope)
            HTTP_REQUESTS.labels(scope.get("method", ""), template, str(status["code"])).inc()
            HTTP_LATENCY.labels(template).observe(time.perf_counter() - started)
