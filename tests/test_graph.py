"""Influence (PageRank), graph queries, graph-enhanced retrieval and the /papers API."""

import asyncio
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from src.agents.graph import ChatDeps, build_graph
from src.agents.graph_retrieval import mastery_level, rerank
from src.api.app import create_app
from src.core.config import settings
from src.graph import store as graph_store
from src.graph.influence import in_degrees, pagerank, refresh_influence
from src.graph.store import InMemoryGraphStore
from tests.chat_fakes import FakeRetriever, RecordingAdapter, hit, make_services, router

# ---------------------------------------------------------------- PageRank


def test_pagerank_rewards_being_cited_by_cited_papers():
    # d and c both cite b; b cites a: a inherits b's importance.
    ranks = pagerank(["a", "b", "c", "d"], [("b", "a"), ("c", "b"), ("d", "b")])
    assert ranks["a"] > ranks["c"] and ranks["b"] > ranks["c"]
    assert ranks["c"] == pytest.approx(ranks["d"])
    assert sum(ranks.values()) == pytest.approx(4.0, rel=1e-3)  # scaled: average is 1.0


def test_pagerank_edge_cases():
    assert pagerank([], []) == {}
    ranks = pagerank(["a", "b"], [("a", "a"), ("a", "zzz")])  # self-loop and unknown node ignored
    assert ranks == {"a": 1.0, "b": 1.0}


def test_in_degree_ignores_duplicates_and_self_citations():
    assert in_degrees(["a", "b"], [("b", "a"), ("b", "a"), ("a", "a")]) == {"a": 1, "b": 0}


# ---------------------------------------------------------------- a small corpus

#   transformer (1706) <- lora (2106) <- qlora (2305) <- new (2401)
#   critic (2402) criticises lora; comparer (2403) compares with lora; external 9999.00001 cited


@pytest.fixture
def graph():
    g = InMemoryGraphStore()
    for pid, title in [
        ("1706.03762v7", "Attention Is All You Need"),
        ("2106.09685v2", "LoRA"),
        ("2305.14314v1", "QLoRA"),
        ("2401.00001v1", "New Paper"),
        ("2402.00002v1", "Critic"),
        ("2403.00003v1", "Comparer"),
    ]:
        g.add_paper(pid, title)
    g.cite("2106.09685v2", "1706.03762")
    g.cite("2305.14314v1", "2106.09685", "builds_on", "We build on LoRA [12].")
    g.cite("2401.00001v1", "2305.14314")
    g.cite("2401.00001v1", "9999.00001")
    g.cite("2402.00002v1", "2106.09685", "critical", "LoRA [3] fails to adapt embeddings.")
    g.cite("2403.00003v1", "2106.09685", "comparative", "We outperform LoRA [4].")
    g.cite("2403.00003v1", "9999.00001")
    return g


async def test_refresh_influence_scores_the_corpus(graph):
    assert await refresh_influence(graph) == 6
    top = await graph.top_influential(2)
    # LoRA is cited three times, but its rank flows on to the Transformer it builds on:
    # the foundational paper ends up most influential - PageRank's whole point.
    assert [t["arxiv_id"] for t in top] == ["1706.03762v7", "2106.09685v2"]
    assert (await graph.paper("2106.09685"))["in_degree"] == 3  # base id resolves too


async def test_prerequisite_chain_follows_citations_with_a_depth_limit(graph):
    chain = await graph.prerequisites("2401.00001v1", max_depth=2, limit=10)
    assert [(p["arxiv_id"], p["depth"]) for p in chain] == [
        ("2305.14314v1", 1),
        ("2106.09685v2", 2),
    ]
    deeper = await graph.prerequisites("2401.00001v1", max_depth=3, limit=10)
    assert deeper[-1]["arxiv_id"] == "1706.03762v7" and deeper[-1]["path"][0] == "2401.00001v1"


async def test_prerequisites_survive_citation_cycles(graph):
    graph.cite("1706.03762v7", "2401.00001")  # a cycle back to the root
    chain = await graph.prerequisites("2401.00001v1", max_depth=4, limit=10)
    assert "2401.00001v1" not in [p["arxiv_id"] for p in chain]


async def test_who_criticises_a_paper(graph):
    rows = await graph.citing_contexts("2106.09685v2", ["critical", "comparative"], 10)
    assert [(r["citing_id"], r["function"]) for r in rows] == [
        ("2402.00002v1", "critical"),
        ("2403.00003v1", "comparative"),
    ]


async def test_missing_papers_ranked_by_how_often_the_corpus_cites_them(graph):
    assert await graph.missing(5) == [{"arxiv_id": "9999.00001", "cited_by": 2}]


def _sql(clause) -> str:
    return str(clause.compile(dialect=postgresql.dialect()))


def test_pg_graph_queries_are_bounded():
    prereq = _sql(graph_store._PREREQUISITES)
    assert "WITH RECURSIVE" in prereq
    assert "NOT (p.arxiv_id = ANY(ch.path))" in prereq  # cycle guard
    assert "ch.depth < %(max_depth)s" in prereq and "LIMIT %(limit)s" in prereq
    assert "AS references" not in _sql(graph_store._PAPER)  # reserved word in Postgres
    assert "NOT EXISTS" in _sql(graph_store._MISSING)


async def test_pg_store_sets_a_statement_timeout(monkeypatch):
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock

    session = AsyncMock()
    session.begin = MagicMock(return_value=AsyncMock())
    session.execute.return_value = MagicMock(__iter__=lambda self: iter([]))

    @asynccontextmanager
    async def fake():
        yield session

    monkeypatch.setattr(graph_store, "get_async_session", fake)
    await graph_store.PgGraphStore(timeout_ms=1500).influence(["x"])
    first = str(session.execute.await_args_list[0].args[0])
    assert first == "SET LOCAL statement_timeout = 1500"


# ---------------------------------------------------------------- re-ranking


def test_mastery_level():
    assert mastery_level({}, {"lora": 0.1, "rlhf": 0.2}) == "weak"
    assert mastery_level({}, {"lora": 0.9}) == "strong"
    assert mastery_level({}, {"lora": 0.5}) is None
    assert mastery_level({"expertise_level": "beginner"}, {}) == "weak"
    assert mastery_level({"expertise_level": "advanced"}, None) == "strong"


def _section_hit(pid, section, published="2020-01-01"):
    h = hit(pid, 0)
    h.section, h.published = section, published
    return h


def test_weak_readers_get_background_and_influential_papers_first():
    hits = [_section_hit("a", "4 Experiments"), _section_hit("b", "1 Introduction")]
    assert [h.paper_id for h in rerank(hits, "weak", {})] == ["b", "a"]
    hits = [_section_hit("a", "3 Method"), _section_hit("b", "3 Method")]
    assert [h.paper_id for h in rerank(hits, "weak", {"b": 4.0})] == ["b", "a"]


def test_strong_readers_get_methods_and_recent_work_first():
    hits = [_section_hit("a", "1 Introduction"), _section_hit("b", "3 Method")]
    assert [h.paper_id for h in rerank(hits, "strong", {})] == ["b", "a"]
    # Recency breaks near-ties further down; it is deliberately too weak to unseat the top hit.
    hits = [_section_hit(f"x{i}", "3 Method", "2015-01-01") for i in range(5)] + [
        _section_hit("old", "3 Method", "2015-01-01"),
        _section_hit("new", "3 Method", "2026-01-01"),
    ]
    ranked = [h.paper_id for h in rerank(hits, "strong", {}, today=date(2026, 9, 1))]
    assert ranked.index("new") < ranked.index("old")
    assert ranked[0] == "x0"


def test_boosts_cannot_leapfrog_a_much_better_hit():
    hits = (
        [_section_hit("best", "4 Experiments")]
        + [_section_hit(f"x{i}", "4 Experiments") for i in range(10)]
        + [_section_hit("far", "1 Introduction")]
    )
    assert rerank(hits, "weak", {})[0].paper_id == "best"


# ---------------------------------------------------------------- graph in the chat loop


def _chat_graph(graph, hits, filtered_only=None):
    adapter = RecordingAdapter()
    retriever = FakeRetriever(hits, filtered_only=filtered_only)
    compiled = build_graph(ChatDeps(llm=router(adapter), retriever=retriever, graph=graph))
    return compiled, adapter


def _input(query, **extra):
    return {"query": query, "history": [], "timings": {}, **extra}


CORPUS_HITS = [
    hit("2401.00001v1", 0),
    hit("2305.14314v1", 0),
    hit("2106.09685v2", 0),
    hit("1706.03762v7", 0),
    hit("2402.00002v1", 0),
]


async def test_weak_learner_gets_prerequisites(graph):
    compiled, adapter = _chat_graph(
        graph, [hit("2401.00001v1", 0)], filtered_only=[hit("2305.14314v1", 0)]
    )
    state = await compiled.ainvoke(
        _input("Explain the new paper", profile={"expertise_level": "beginner"})
    )
    roles = {s["paper_id"]: s["role"] for s in state["sources"]}
    assert roles["2401.00001v1"] == "retrieved"
    assert roles.get("2305.14314v1") == "prerequisite"
    assert state["graph_status"] == "ok"
    assert "(prerequisite)" in adapter.requests[0].messages[0].content


async def test_critique_brings_in_what_other_papers_say(graph):
    compiled, adapter = _chat_graph(graph, [hit("2106.09685v2", 0)])
    # (not "What are the weaknesses...": that ties learn/critique and falls back to chat)
    state = await compiled.ainvoke(_input("Critique the weaknesses and baselines of LoRA"))
    assert state["intent"] == "critique"
    ctx = [s for s in state["sources"] if s["role"] == "citation_context"]
    assert [c["content"] for c in ctx] == [
        "LoRA [3] fails to adapt embeddings.",
        "We outperform LoRA [4].",
    ]
    assert ctx[0]["paper_id"] == "2402.00002v1" and ctx[0]["section"] == "on Title 2106.09685v2"
    assert "(what another paper says)" in adapter.requests[0].messages[0].content


async def test_synthesis_adds_connected_papers(graph):
    # Only the new paper is retrieved; QLoRA (which it cites) is reachable through the graph.
    compiled, _ = _chat_graph(
        graph,
        [hit("2401.00001v1", 0), hit("2305.14314v1", 0)],
        filtered_only=[hit("2106.09685v2", 0)],
    )
    state = await compiled.ainvoke(_input("Compare the new paper with QLoRA"))
    assert any(s["role"] == "related" for s in state["sources"])


async def test_graph_failure_falls_back_to_vector_only(graph):
    async def broken(*args, **kwargs):
        raise ConnectionError("postgres down")

    graph.influence = broken
    compiled, _ = _chat_graph(graph, [hit("2106.09685v2", 0)])
    state = await compiled.ainvoke(_input("What are the weaknesses of LoRA?"))
    assert state["graph_status"] == "failed"
    assert [s["role"] for s in state["sources"]] == ["retrieved"]
    assert state["response"]  # the answer still happens


async def test_slow_graph_is_cut_off_by_the_timeout(graph, monkeypatch):
    monkeypatch.setattr(settings, "GRAPH_TIMEOUT_SECONDS", 0.05)

    async def slow(*args, **kwargs):
        await asyncio.sleep(1)
        return {}

    graph.influence = slow
    compiled, _ = _chat_graph(graph, [hit("2106.09685v2", 0)])
    state = await compiled.ainvoke(_input("Explain LoRA"))
    assert state["graph_status"] == "failed" and state["timings"]["retrieve"] < 500


async def test_no_graph_configured_is_off():
    compiled, _ = _chat_graph(None, [hit("p", 0)])
    assert (await compiled.ainvoke(_input("Explain LoRA")))["graph_status"] == "off"


# ---------------------------------------------------------------- /papers API


@pytest.fixture
def api(graph):
    asyncio.run(refresh_influence(graph))
    graph.add_paper("hep-th/9802150v1", "Old style id")
    with TestClient(create_app(services=make_services(graph_store=graph))) as client:
        yield client


def test_paper_endpoints(api):
    assert api.get("/api/v1/papers/2106.09685v2").json()["title"] == "LoRA"
    assert api.get("/api/v1/papers/hep-th/9802150v1").json()["title"] == "Old style id"
    assert api.get("/api/v1/papers/0000.00000").status_code == 404
    assert api.get("/api/v1/papers/top?limit=1").json()[0]["arxiv_id"] == "1706.03762v7"
    assert api.get("/api/v1/papers/missing").json()[0]["arxiv_id"] == "9999.00001"


def test_graph_prerequisites_and_citations_endpoints(api):
    data = api.get("/api/v1/papers/2106.09685v2/graph").json()
    ids = {n["id"] for n in data["nodes"]}
    assert {"2106.09685v2", "1706.03762v7", "2402.00002v1"} <= ids
    chain = api.get("/api/v1/papers/2401.00001v1/prerequisites?depth=2").json()
    assert [p["arxiv_id"] for p in chain] == ["2305.14314v1", "2106.09685v2"]
    critical = api.get("/api/v1/papers/2106.09685v2/citations?function=critical").json()
    assert [c["citing_id"] for c in critical] == ["2402.00002v1"]
    assert api.get("/api/v1/papers/2106.09685v2/graph?depth=5").status_code == 400


def test_public_endpoints_are_rate_limited_per_ip(api, monkeypatch):
    api.app.state.services.public_limiter.limit = 2
    codes = [api.get("/api/v1/papers/top").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_graph_viewer_page(api):
    r = api.get("/ui/graph")
    assert r.status_code == 200 and "forceSimulation" in r.text
