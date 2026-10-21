"""Phase 5 building blocks: trailer, checks, prompt registry, tutor logic, advisor ranking."""

import pytest
from jinja2 import TemplateNotFound, UndefinedError

from src.agents import advisor, prompts, tutor
from src.agents.checks import (
    critic_checks,
    last_question,
    questions,
    synthesizer_checks,
    tutor_checks,
)
from src.agents.intent import IntentResult
from src.agents.trailer import DELIMITER, TrailerSplitter, parse_meta, split_complete
from src.core.config import settings
from src.memory.model import signals_for

# ---------------------------------------------------------------- trailer


def _stream(chunks):
    s = TrailerSplitter()
    shown = "".join(s.feed(c) for c in chunks) + s.close()
    return shown, s.meta_text


def test_delimiter_split_across_tokens_never_leaks():
    shown, meta = _stream(["Explain. What is x?\n<<<ME", "TA>>>", '\n{"question": "What is x?"}'])
    assert shown.strip() == "Explain. What is x?"
    assert parse_meta(meta) == {"question": "What is x?"}


def test_text_that_only_looks_like_a_delimiter_start_is_released():
    shown, meta = _stream(["a <<< b", " c <<"])
    assert shown == "a <<< b c <<" and meta == ""


def test_no_trailer_means_no_meta():
    assert split_complete("Just prose.") == ("Just prose.", None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"verdict": "correct"}\n```', {"verdict": "correct"}),
        ('Sure! {"a": 1} hope that helps', {"a": 1}),
        ("{not json}", None),
        ("[1, 2]", None),
    ],
)
def test_parse_meta_is_tolerant(raw, expected):
    assert parse_meta(raw) == expected


# ---------------------------------------------------------------- checks


def test_questions_ignore_math_and_code():
    text = "Consider $P(x|y)?$ and `a?b`. Why does it work? Explain."
    assert questions(text) == ["Why does it work?"]
    assert last_question("One? Two?") == "Two?"


def test_tutor_checks():
    assert tutor_checks("Idea. Why?", {"q": 1}, expects_question=True) == {
        "trailer_parsed": True,
        "one_question": True,
    }
    assert tutor_checks("Why? How?", None, True)["one_question"] is False
    assert tutor_checks("Done.", {}, expects_question=False)["no_question"] is True


def test_critic_checks():
    good = "\n".join(
        f"## {s}\n- point [1]"
        for s in (
            "Summary",
            "Strengths",
            "Weaknesses",
            "Assumptions",
            "Baselines and evaluation",
            "Reproducibility",
            "Open questions",
        )
    )
    assert critic_checks(good) == {"all_sections": True, "claims_cited": True}
    assert critic_checks(good.replace("- point [1]", "- uncited", 3))["claims_cited"] is False
    assert critic_checks("## Summary\nx")["all_sections"] is False


def test_synthesizer_checks():
    text = (
        "## Comparison\n| Paper | Approach |\n|---|---|\n| A [1] | x |\n"
        "## Where they agree\n## Where they disagree\n## Timeline\n## Open directions\n"
    )
    assert synthesizer_checks(text, {"p1", "p2"}) == {
        "all_sections": True,
        "comparison_table": True,
        "two_or_more_papers": True,
    }
    assert synthesizer_checks("no table", {"p1"})["two_or_more_papers"] is False


# ---------------------------------------------------------------- prompts


BASE = {"reader": [], "pitch": "standard", "max_words": 200}
SAMPLE_VARS = {
    "general": {"reader": []},
    "critic": {"reader": [], "focus": "the work"},
    "synthesizer": {"reader": [], "topic": "t"},
    "tutor_explain": {**BASE, "topic": "t", "math": True},
    "tutor_grade": {
        **BASE,
        "question": "q?",
        "answer_key": "k",
        "answer": "a",
        "hints_left": True,
        "is_last": False,
    },
    "tutor_hint": {**BASE, "question": "q?", "answer_key": "k", "hints_used": 1},
    "tutor_reexplain": {**BASE, "topic": "t", "previous": "p"},
    "tutor_review": {**BASE, "topic": "t"},
    "tutor_reading": {
        **BASE,
        "paper_title": "P",
        "paper_id": "1",
        "section": "S",
        "index": 0,
        "total": 3,
        "is_first": True,
        "is_last": False,
        "prerequisites": ["X"],
    },
    "tutor_equation": {
        **BASE,
        "paper_title": "P",
        "paper_id": "1",
        "section": "S",
        "latex": r"\frac{a}{b}",
        "number": "(3)",
    },
}


def test_every_template_compiles_and_renders():
    versions = prompts.validate_templates()
    assert set(versions) == set(SAMPLE_VARS)
    for name, variables in SAMPLE_VARS.items():
        for version in versions[name]:
            text = prompts._env.get_template(f"{name}.{version}.j2").render(**variables)
            assert "Answer ONLY from the numbered sources" in text
            if name.startswith("tutor_"):
                assert DELIMITER in text  # the trailer contract is in every tutor prompt


def test_missing_variables_fail_loudly():
    with pytest.raises(UndefinedError):
        prompts.render("tutor_explain", None, reader=[])


def test_misconfigured_variant_stops_startup(monkeypatch):
    monkeypatch.setattr(settings, "PROMPT_VARIANTS", {"tutor_explain": ["v1", "v9"]})
    with pytest.raises(TemplateNotFound):
        prompts.validate_templates()


def test_ab_split_is_stable_per_user_and_uses_both_variants(monkeypatch):
    monkeypatch.setattr(settings, "PROMPT_AB_TEST", True)
    picks = {uid: prompts.choose_version("tutor_explain", uid) for uid in range(1, 201)}
    assert set(picks.values()) == {"v1", "v2"}
    assert 60 <= sum(v == "v1" for v in picks.values()) <= 140  # roughly half
    assert all(prompts.choose_version("tutor_explain", u) == v for u, v in picks.items())


def test_ab_off_or_anonymous_gets_the_default(monkeypatch):
    assert prompts.choose_version("tutor_explain", None) == "v1"
    monkeypatch.setattr(settings, "PROMPT_AB_TEST", False)
    assert prompts.choose_version("tutor_explain", 7) == "v1"


def test_render_reports_the_version():
    _, version = prompts.render("critic", None, reader=[], focus="x")
    assert version == "critic@v1"


# ---------------------------------------------------------------- tutor: detecting actions

CHAT = IntentResult("chat", 0.3, "rules")
LEARN = IntentResult("learn", 1.0, "rules")
PENDING = {"concept": "lora", "pending": {"question": "Why low rank?", "answer_key": "k"}}


@pytest.mark.parametrize(
    ("message", "state", "expected"),
    [
        ("Because the update is small", PENDING, "grade"),
        ("hint please", PENDING, "hint"),
        ("can I get a hint", PENDING, "hint"),
        ("I don't understand", PENDING, "reexplain"),
        ("this is too complicated", {"topic": "x"}, "reexplain"),
        ("skip", PENDING, "skip"),
        ("I give up", PENDING, "skip"),
        ("next", {"reading": {"paper_id": "p", "index": 0}}, "read_next"),
        ("quiz me", None, "review"),
        ("walk me through 2106.09685", None, "read"),
        ("explain equation (3) of 2106.09685v2", None, "equation"),
        ("hint", None, None),  # nothing pending: not a tutor command
        ("I don't understand attention", None, None),  # no session: a normal learn question
    ],
)
def test_detect(message, state, expected):
    action, _ = tutor.detect(message, state, None, None, CHAT, [])
    assert action == expected


def test_a_new_question_interrupts_a_pending_one():
    action, _ = tutor.detect("What is RLHF?", PENDING, None, None, LEARN, ["rlhf"])
    assert action is None


def test_a_hesitant_answer_is_still_an_answer():
    action, _ = tutor.detect("is it because rank is low?", PENDING, None, None, CHAT, [])
    assert action == "grade"


def test_explicit_actions():
    assert tutor.detect("x", PENDING, "answer", None, CHAT, [])[0] == "grade"
    assert tutor.detect("x", None, "answer", None, CHAT, [])[0] is None  # nothing to answer
    assert tutor.detect("x", PENDING, "dont_understand", None, CHAT, [])[0] == "reexplain"
    action, args = tutor.detect(
        "go", None, "equation", {"paper_id": "p", "equation": "2"}, CHAT, []
    )
    assert (action, args) == ("equation", {"paper_id": "p", "equation": "2"})


def test_arguments_are_extracted_from_text():
    _, args = tutor.detect("explain eq. 12 of 2305.14314", None, None, None, CHAT, [])
    assert args == {"paper_id": "2305.14314", "equation": "12"}


# ---------------------------------------------------------------- tutor: transitions

SRC = [{"n": 1, "content": "x" * 5000, "paper_id": "p", "title": "T", "section": "S", "url": "u"}]


def _t(prev, action, meta=None, text="", **kw):
    kw.setdefault("topic", "LoRA")
    kw.setdefault("concept", "lora")
    kw.setdefault("sources", SRC)
    return tutor.transition(prev, action, meta=meta, text=text, **kw)


def test_explain_starts_a_session_with_a_pending_question():
    ts, events, _ = _t(None, "explain", {"question": "Why?", "answer_key": "Because."})
    assert ts["pending"] == {"question": "Why?", "answer_key": "Because."}
    assert (ts["step"], ts["level"], ts["concept"]) == (1, 0, "lora")
    assert len(ts["sources"][0]["content"]) == settings.TUTOR_STATE_SOURCE_CHARS
    assert events == []


def test_missing_trailer_is_repaired_from_the_prose():
    ts, _, _ = _t(None, "explain", None, text="Idea. So, why low rank? ")
    assert ts["pending"]["question"] == "So, why low rank?"


def test_correct_answer_advances_and_is_evidence():
    ts, _, _ = _t(None, "explain", {"question": "Q1?", "answer_key": "K1"})
    ts, events, verdict = _t(
        ts, "grade", {"verdict": "correct", "question": "Q2?", "answer_key": "K2"}
    )
    assert verdict == "correct" and ts["step"] == 2
    assert ts["pending"] == {"question": "Q2?", "answer_key": "K2"}
    assert events == [{"kind": "quiz", "payload": {"concept": "lora", "result": "correct"}}]


def test_wrong_answers_use_hints_then_step_down(monkeypatch):
    monkeypatch.setattr(settings, "TUTOR_MAX_HINTS", 1)
    ts, _, _ = _t(None, "explain", {"question": "Q1?", "answer_key": "K1"})
    ts, events, _ = _t(ts, "grade", {"verdict": "incorrect", "question": "Q1?"})
    assert ts["hints_used"] == 1 and ts["pending"]["answer_key"] == "K1"  # key kept on re-ask
    assert events[0]["payload"]["result"] == "incorrect"
    ts, _, _ = _t(ts, "grade", {"verdict": "partial", "question": "Easier?", "answer_key": "E"})
    assert (ts["level"], ts["hints_used"]) == (-1, 0)  # hints exhausted: simpler level
    assert ts["pending"]["question"] == "Easier?"


def test_final_correct_answer_ends_the_session():
    ts, _, _ = _t(None, "explain", {"question": "Q?", "answer_key": "K"})
    ts, _, _ = _t(ts, "grade", {"verdict": "correct", "question": None})
    assert ts["pending"] is None


def test_reexplain_steps_down_and_records_confusion():
    ts, _, _ = _t(None, "explain", {"question": "Q?"})
    ts, events, _ = _t(ts, "reexplain", {"question": "Easy?"})
    assert ts["level"] == -1 and ts["pending"]["question"] == "Easy?"
    assert events == [{"kind": "feedback", "payload": {"signal": "confused", "concepts": ["lora"]}}]
    for _ in range(3):
        ts, _, _ = _t(ts, "reexplain", {"question": "Easier?"})
    assert ts["level"] == -2  # floor


def test_hint_and_skip():
    ts, _, _ = _t(None, "explain", {"question": "Q?", "answer_key": "K"})
    ts, _, _ = _t(ts, "hint", {"question": "Q again?"})
    assert ts["hints_used"] == 1 and ts["pending"] == {"question": "Q again?", "answer_key": "K"}
    ts, events, _ = _t(ts, "skip")
    assert ts["pending"] is None and events == []


def test_unknown_verdict_records_nothing():
    ts, _, _ = _t(None, "explain", {"question": "Q?"})
    _, events, verdict = _t(ts, "grade", {"verdict": "maybe"})
    assert verdict is None and events == []


def test_concept_from_free_text_is_mapped_to_the_vocabulary():
    ts, _, _ = _t(
        None, "explain", {"question": "Q?", "concept": "Low-Rank Adaptation"}, concept=None
    )
    assert ts["concept"] == "lora"


def test_public_view_hides_answer_keys_and_sources():
    ts, _, _ = _t(None, "explain", {"question": "Q?", "answer_key": "SECRET"})
    view = tutor.public_view(ts)
    assert view["pending_question"] == "Q?"
    assert "SECRET" not in str(view) and "sources" not in view


def test_pitch_steps_down_with_level():
    assert tutor.pitch("strong", 0).startswith("advanced")
    assert tutor.pitch("strong", -2).startswith("simple:")
    assert tutor.pitch("weak", -2).startswith("simplest")


def test_partial_quiz_is_a_signal():
    assert signals_for("quiz", {"concept": "lora", "result": "partial"}) == [
        ("lora", "quiz_partial")
    ]
    assert signals_for("quiz", {"concept": "lora", "correct": True}) == [("lora", "quiz_correct")]


# ---------------------------------------------------------------- advisor


def _cand(pid, rank, concepts=(), prereq_of=(), pagerank=0.0):
    return advisor.Candidate(
        pid,
        f"Title {pid}",
        "snippet",
        rank,
        concepts=list(concepts),
        prerequisite_of=list(prereq_of),
        pagerank=pagerank,
    )


def test_advisor_prefers_foundations_and_gaps_with_reasons():
    ranked = advisor.rank_candidates(
        [
            _cand("top", 0),
            _cand("gap", 1, concepts=["lora"]),
            _cand("found", 2, prereq_of=["A", "B"], pagerank=3.0),
        ],
        weak={"lora"},
        interests=set(),
        limit=3,
    )
    assert [c.paper_id for c in ranked][:2] == ["found", "gap"]
    assert "foundational: A, B build on it" in ranked[0].reasons
    assert any("still building" in r for r in ranked[1].reasons)


def test_advisor_limits_and_explains_every_pick():
    ranked = advisor.rank_candidates([_cand(str(i), i) for i in range(10)], set(), set(), 3)
    assert len(ranked) == 3 and all(c.reasons for c in ranked)


def test_advisor_reply_cites_and_mentions_due_reviews():
    ranked = advisor.rank_candidates([_cand("p", 0)], set(), set(), 1)
    text = advisor.render(ranked, due=["lora"], weak=["rlhf"])
    assert "**Title p** (p)" in text and "[1]" in text
    assert "due for review: LoRA" in text and "quiz me" in text


def test_a_question_keeps_its_maths():
    """Found in the browser: the tutor's exercise read "if   was increased to be closer to  ?"
    because the fallback question was taken from text with the formulas blanked out."""
    text = r"Idea first. How would the parameters change if $r$ approaches $\min(d, k)$?"
    assert questions(text) == [r"How would the parameters change if $r$ approaches $\min(d, k)$?"]
    assert last_question(text) == questions(text)[0]


def test_a_question_mark_inside_a_formula_still_does_not_count_or_split():
    text = "We write $p(y?x)$ here. Why is $0.5$ fair? Say more."
    assert questions(text) == ["Why is $0.5$ fair?"]


@pytest.mark.parametrize("placeholder", ["N/A", "n/a", "None", "null", " - ", "No question.", ""])
def test_a_placeholder_question_does_not_become_an_exercise(placeholder):
    """Found in the browser: with nothing in the sources to ask about, the model wrote "N/A" as the
    question and the page showed an Exercise titled N/A."""
    state, _, _ = tutor.transition(
        None,
        "explain",
        meta={"question": placeholder, "answer_key": "N/A"},
        text="The sources do not cover this topic.",
        topic="x",
        concept=None,
        sources=[],
    )
    assert not (state or {}).get("pending")


def test_a_real_question_is_kept_and_stripped():
    state, _, _ = tutor.transition(
        None,
        "explain",
        meta={"question": "  Why does $r$ matter?  ", "answer_key": "low rank"},
        text="Idea.",
        topic="x",
        concept=None,
        sources=[],
    )
    assert state["pending"] == {"question": "Why does $r$ matter?", "answer_key": "low rank"}
