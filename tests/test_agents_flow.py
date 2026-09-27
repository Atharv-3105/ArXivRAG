"""The specialists end to end through the API, over the offline mock LLM (which follows every
specialist's output contract, and grades tutor answers by overlap with the answer key)."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.agents.specialists import NOT_FOUND, NOTHING_DUE, READING_DONE
from src.agents.trailer import DELIMITER
from src.api.app import create_app
from src.core.config import settings
from src.storage.paper_content import InMemoryPaperContent
from tests.chat_fakes import ScriptedAdapter, hit, make_services, signup

LORA = "LoRA freezes the pretrained weights and trains low-rank update matrices."
HITS = [
    hit("2106.09685v2", 0, content=LORA),
    hit("2305.14314v1", 0, content="QLoRA quantizes the frozen model to 4-bit NF4."),
    hit("1706.03762v7", 0, content="The Transformer relies entirely on attention."),
]


@pytest.fixture
def content():
    c = InMemoryPaperContent()
    c.add(
        "2106.09685v2",
        "LoRA",
        [
            {"title": "Abstract", "chunks": ["We propose LoRA, low-rank adaptation."]},
            {
                "title": "4 Method",
                "path": "4 Method",
                "chunks": ["The update is $\\Delta W = BA$."],
            },
        ],
        equations=[{"latex": r"h = W_0 x + B A x", "number": "(3)", "section_path": "4 Method"}],
    )
    return c


@pytest.fixture
def services(content):
    return make_services(HITS, content=content)


@pytest.fixture
def api(services):
    with TestClient(create_app(services=services)) as client:
        client.headers.update(signup(client))
        yield client


def chat(api, message, cid=None, **extra):
    body = {"message": message, **extra}
    if cid:
        body["conversation_id"] = cid
    r = api.post("/api/v1/chat", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def drain(api, services):
    api.portal.call(services.memory.drain)


# ---------------------------------------------------------------- routing


@pytest.mark.parametrize(
    ("message", "agent"),
    [
        ("Explain how LoRA works", "tutor"),
        ("Critique the baselines and weaknesses of LoRA", "critic"),
        ("Compare LoRA and QLoRA", "synthesizer"),
        ("What should I read next?", "advisor"),
        ("LoRA", "general"),
    ],
)
def test_each_intent_reaches_its_specialist(api, message, agent):
    assert chat(api, message)["agent"] == agent


# ---------------------------------------------------------------- tutor


def test_full_tutoring_loop_right_answer(api, services):
    first = chat(api, "Explain how LoRA works")
    assert first["action"] == "explain"
    tutor = first["tutor"]
    assert tutor["pending_question"].endswith("?") and tutor["step"] == 1
    assert DELIMITER not in first["answer"]  # the trailer never reaches the reader
    assert first["checks"]["one_question"] and first["checks"]["trailer_parsed"]
    assert first["prompt_version"].startswith("tutor_explain@v")

    # The mock grades by overlap with the answer key (the first sentence of source [1]).
    second = chat(api, LORA, first["conversation_id"])
    assert (second["action"], second["verdict"]) == ("grade", "correct")
    drain(api, services)
    [row] = api.get("/api/v1/me/mastery").json()
    assert row["concept"] == "lora" and row["evidence"] == 2  # asked + answered


def test_wrong_answer_gets_a_hint_and_the_question_again(api):
    first = chat(api, "Explain how LoRA works")
    cid = first["conversation_id"]
    wrong = chat(api, "bananas grow on trees", cid)
    assert wrong["verdict"] == "incorrect"
    assert wrong["tutor"]["hints_used"] == 1
    assert wrong["tutor"]["pending_question"] == first["tutor"]["pending_question"]
    hint = chat(api, "hint", cid)
    assert hint["action"] == "hint" and hint["tutor"]["hints_used"] == 2
    assert "Hint" in hint["answer"]


def test_i_dont_understand_steps_down_and_counts_as_confusion(api, services):
    first = chat(api, "Explain how LoRA works")
    again = chat(api, "I don't understand", first["conversation_id"])
    assert again["action"] == "reexplain" and again["tutor"]["level"] == -1
    drain(api, services)
    audit = api.get("/api/v1/me/audit").json()
    assert any(a["field"] == "mastery:lora" for a in audit)


def test_explicit_button_action(api):
    first = chat(api, "Explain how LoRA works")
    r = chat(api, "I'm lost", first["conversation_id"], action="dont_understand")
    assert r["action"] == "reexplain"


def test_skip_reveals_the_answer_and_closes_the_question(api, services):
    first = chat(api, "Explain how LoRA works")
    skipped = chat(api, "skip", first["conversation_id"])
    assert skipped["provider"] == "none" and "For reference" in skipped["answer"]
    assert skipped["tutor"]["pending_question"] is None


def test_new_question_while_a_question_is_pending(api):
    first = chat(api, "Explain how LoRA works")
    r = chat(api, "Compare QLoRA and LoRA quantization?", first["conversation_id"])
    assert r["agent"] == "synthesizer"


def test_session_state_survives_between_requests(api, services):
    first = chat(api, "Explain how LoRA works")
    cid = first["conversation_id"]
    stored = services.conversations.states[cid]
    assert stored["pending"]["answer_key"]  # kept server side, never shown
    messages = api.get(f"/api/v1/conversations/{cid}").json()
    assert messages[1]["agent"] == "tutor" and messages[1]["prompt_version"]


def test_scripted_model_without_trailer_is_repaired(services):
    services.llm.slots[0].adapter = ScriptedAdapter("LoRA is low-rank [1]. Why low rank?")
    with TestClient(create_app(services=services)) as client:
        client.headers.update(signup(client))
        r = chat(client, "Explain how LoRA works")
    assert r["tutor"]["pending_question"] == "Why low rank?"
    assert r["checks"]["trailer_parsed"] is False and r["checks"]["one_question"] is True


def test_stream_never_leaks_the_trailer(api):
    r = api.post("/api/v1/chat/stream", json={"message": "Explain how LoRA works"})
    blocks = [b for b in r.text.strip().split("\n\n")]
    events = [dict(line.split(": ", 1) for line in b.splitlines()) for b in blocks]
    tokens = "".join(json.loads(e["data"])["text"] for e in events if e["event"] == "token")
    assert DELIMITER not in tokens and "answer_key" not in tokens
    intent = json.loads(next(e["data"] for e in events if e["event"] == "intent"))
    assert (intent["agent"], intent["action"]) == ("tutor", "explain")
    done = json.loads(events[-1]["data"])
    assert done["tutor"]["pending_question"] and done["prompt_version"]


# ---------------------------------------------------------------- review


def test_review_quizzes_a_due_concept(api, services):
    assert chat(api, "quiz me")["answer"] == NOTHING_DUE
    chat(api, "Explain how LoRA works")
    drain(api, services)
    services.profiles.mastery[1]["lora"].next_review_at = datetime.now(UTC) - timedelta(hours=1)
    r = chat(api, "quiz me")
    assert (r["action"], r["tutor"]["mode"], r["tutor"]["concept"]) == ("review", "review", "lora")
    assert r["tutor"]["pending_question"]


# ---------------------------------------------------------------- guided reading & equations


def test_guided_reading_walks_the_sections_in_order(api):
    first = chat(api, "walk me through 2106.09685v2")
    assert first["action"] == "read"
    assert first["tutor"]["reading"] == {
        "paper_id": "2106.09685v2",
        "title": "LoRA",
        "index": 0,
        "total": 2,
        "section": "Abstract",
    }
    cid = first["conversation_id"]
    second = chat(api, "next", cid)
    assert second["tutor"]["reading"]["index"] == 1
    assert second["sources"][0]["role"] == "reading"
    done = chat(api, "next", cid)
    assert done["answer"] == READING_DONE and done["tutor"]["reading"] is None


def test_equation_walkthrough_uses_the_stored_latex(api, services):
    r = chat(api, "explain equation 3 of 2106.09685v2")
    assert r["action"] == "equation" and r["tutor"]["mode"] == "equation"
    system = services.llm.slots[0].adapter.requests[-1].messages[0].content
    assert "h = W_0 x + B A x" in system and "Every symbol" in system


def test_unknown_paper_or_equation(api):
    assert chat(api, "walk me through 9999.99999")["answer"] == NOT_FOUND
    assert chat(api, "explain equation 42 of 2106.09685v2")["answer"] == NOT_FOUND


# ---------------------------------------------------------------- critic, synthesizer, advisor


def test_critic_follows_its_contract(api, services):
    r = chat(api, "Critique the baselines and weaknesses of LoRA")
    assert r["checks"] == {"all_sections": True, "claims_cited": True, "citations_grounded": True}
    assert services.llm.slots[0].adapter.requests[-1].task == "critic"


def test_synthesizer_follows_its_contract_and_retrieves_broadly(api, services):
    r = chat(api, "Compare LoRA and QLoRA")
    assert r["checks"]["comparison_table"] and r["checks"]["two_or_more_papers"]
    assert services.retriever.queries[-1].startswith("Compare")


def test_advisor_ranks_without_an_llm_call(api, services):
    services.llm.slots[0].adapter.requests.clear()
    r = chat(api, "What should I read next?")
    assert services.llm.slots[0].adapter.requests == []  # deterministic ranking
    assert r["provider"] == "advisor" and r["prompt_version"] == "advisor@rules-v1"
    assert r["citations"] and r["sources"][0]["role"] == "recommendation"
    assert "Here is what I'd read next" in r["answer"]


def test_ab_variant_is_recorded(api, services, monkeypatch):
    monkeypatch.setattr(settings, "PROMPT_AB_TEST", True)
    version = chat(api, "Explain how LoRA works")["prompt_version"]
    assert version in ("tutor_explain@v1", "tutor_explain@v2")
