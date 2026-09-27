"""Endpoints the web app needs, serving the built app, and the offline demo server."""

import pytest
from fastapi.testclient import TestClient

from src.api import demo
from src.api.app import create_app
from src.core.config import settings
from src.graph.influence import refresh_influence
from src.graph.store import InMemoryGraphStore
from src.memory import concepts
from tests.chat_fakes import hit, make_services, signup


@pytest.fixture
def api():
    graph = InMemoryGraphStore()
    graph.add_paper("1706.03762v7", "Attention Is All You Need", "2017-06-12", "cs.CL")
    graph.add_paper("2106.09685v2", "LoRA", "2021-06-17", "cs.CL")
    graph.add_paper("2305.14314v1", "QLoRA", "2023-05-23", "cs.LG")
    graph.cite("2106.09685v2", "1706.03762", "builds_on", "LoRA adapts the Transformer [1].")
    graph.cite("2305.14314v1", "2106.09685", "critical", "LoRA [2] still needs 16-bit weights.")
    graph.cite("2305.14314v1", "9999.00001")  # not in the library: not on the map
    services = make_services([hit("2106.09685v2", 0)], graph_store=graph)
    with TestClient(create_app(services=services)) as client:
        client.portal.call(refresh_influence, graph)
        client.headers.update(signup(client))
        yield client


def test_map_returns_places_and_trails_with_their_function(api):
    data = api.get("/api/v1/papers/map").json()
    assert {n["id"] for n in data["nodes"]} == {"1706.03762v7", "2106.09685v2", "2305.14314v1"}
    assert data["nodes"][0]["id"] == "1706.03762v7"  # most influential first
    assert data["nodes"][0]["category"] == "cs.CL"
    edges = {(e["source"], e["target"]): e["function"] for e in data["edges"]}
    assert edges == {
        ("2106.09685v2", "1706.03762v7"): "builds_on",
        ("2305.14314v1", "2106.09685v2"): "critical",
    }


def test_studied_papers_come_from_cited_answers(api):
    assert api.get("/api/v1/me/papers").json() == []
    api.post("/api/v1/chat", json={"message": "Explain how LoRA works"})
    [row] = api.get("/api/v1/me/papers").json()
    assert row["paper_id"] == "2106.09685v2" and row["times"] >= 1


def test_tutor_state_of_a_conversation(api):
    cid = api.post("/api/v1/chat", json={"message": "Explain how LoRA works"}).json()[
        "conversation_id"
    ]
    tutor = api.get(f"/api/v1/conversations/{cid}/tutor").json()["tutor"]
    assert tutor["pending_question"] and tutor["mode"] == "explain"
    other = signup(api, "someone@example.com")
    assert api.get(f"/api/v1/conversations/{cid}/tutor", headers=other).status_code == 404


# ---------------------------------------------------------------- serving the built app


@pytest.fixture
def web(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>ArXivMind</title>", encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("do not serve", encoding="utf-8")
    monkeypatch.setattr(settings, "FRONTEND_DIST", str(dist))
    with TestClient(create_app(services=make_services())) as client:
        yield client


def test_app_shell_is_served_with_security_headers(web):
    r = web.get("/")
    assert r.status_code == 200 and "<title>ArXivMind</title>" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-cache"


def test_client_routes_fall_back_to_the_app_shell(web):
    for path in ("/chat/123", "/map", "/reviews", "/me"):
        assert "<title>ArXivMind</title>" in web.get(path).text


def test_hashed_assets_are_cached_forever(web):
    r = web.get("/assets/index-abc123.js")
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]
    assert web.get("/favicon.svg").text == "<svg/>"


def test_api_paths_never_fall_back_to_html(web):
    r = web.get("/api/v1/does-not-exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    assert web.get("/health/live").json()["status"] == "healthy"


def test_files_outside_dist_are_not_served(web):
    r = web.get("/..%2Fsecret.txt")
    assert "do not serve" not in r.text


def test_api_only_when_the_app_is_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FRONTEND_DIST", str(tmp_path / "missing"))
    with TestClient(create_app(services=make_services())) as client:
        assert client.get("/").status_code == 404


# ---------------------------------------------------------------- demo server and labels


async def test_demo_services_are_complete_and_searchable():
    services = await demo.build_services()
    hits = await services.retriever.search("Compare RLHF and DPO")
    titles = {h.title for h in hits}
    assert any("Direct Preference Optimization" in t for t in titles)  # alias expansion
    graph = await services.graph_store.map(50)
    assert len(graph["nodes"]) == len(demo.PAPERS) and graph["edges"]
    assert await services.content.equation("2106.09685v2", "3")


@pytest.mark.parametrize(
    ("concept", "name"),
    [
        ("lora", "LoRA"),
        ("rlhf", "RLHF"),
        ("self_attention", "Self-attention"),
        ("unknown_x", "Unknown x"),
    ],
)
def test_concept_display_names(concept, name):
    assert concepts.label(concept) == name
