"""The learner model (pure functions) and the concept vocabulary."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from src.memory.concepts import extract_concepts
from src.memory.model import (
    DEFAULT_PROFILE,
    PRIOR,
    MasteryState,
    ProfileState,
    Snapshot,
    compute_update,
    decayed,
    effective_mastery,
    ema,
    explicit_profile_changes,
    sm2,
)

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)
ALPHA, HALF_LIFE = 0.3, 30.0


def _snapshot(mastery=None, profile=None, locked=None) -> Snapshot:
    return Snapshot(
        ProfileState({**DEFAULT_PROFILE, **(profile or {})}, locked or [], 0, NOW),
        {m.concept: m for m in (mastery or [])},
    )


def _update(snapshot, kind, payload, now=NOW):
    return compute_update(snapshot, kind, payload, now, ALPHA, HALF_LIFE)


# ---------- concepts


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How does LoRA compare to low-rank adaptation with QLoRA?", ["lora", "qlora"]),
        ("Explain multi-head attention", ["multi_head_attention"]),  # longest alias wins
        ("RLHF vs DPO for alignment", ["rlhf", "dpo", "alignment"]),
        ("gradient clipping and warmup", ["learning_rate_schedule"]),  # "clipping" is not "clip"
        ("the weather is nice", []),
    ],
)
def test_concepts_map_aliases_to_one_stable_slug(text, expected):
    assert extract_concepts(text) == expected


def test_concept_extraction_is_capped():
    text = "transformer attention lora qlora rlhf dpo moe rag vae gan"
    assert len(extract_concepts(text, limit=4)) == 4


# ---------- primitives


def test_ema_moves_toward_the_observation_and_clamps():
    assert ema(0.5, 1.0, 1.0, 0.3) == pytest.approx(0.65)
    assert ema(0.99, 1.0, 1.0, 1.0) == 1.0
    assert ema(0.0, -5.0, 1.0, 1.0) == 0.0  # clamped to [0, 1]


def test_decay_halves_the_distance_to_the_prior_each_half_life():
    assert decayed(0.9, 0.5, NOW - timedelta(days=30), NOW, 30) == pytest.approx(0.7)
    assert decayed(0.9, 0.5, NOW, NOW, 30) == pytest.approx(0.9)


def test_locked_mastery_never_decays():
    state = MasteryState("lora", 0.9, NOW - timedelta(days=365), locked=True)
    assert effective_mastery(state, 0.2, NOW, HALF_LIFE) == 0.9


def test_sm2_interval_ladder_and_reset():
    s = MasteryState("lora", 0.5, NOW)
    s = sm2(s, 5, NOW)
    assert s.interval_days == 1 and s.next_review_at == NOW + timedelta(days=1)
    second = sm2(s, 5, NOW)
    assert second.interval_days == 6
    s = sm2(second, 5, NOW)
    assert s.interval_days == pytest.approx(6 * second.ease, abs=0.01)  # then x ease
    failed = sm2(s, 1, NOW)
    assert (failed.interval_days, failed.reviews) == (1, 0)  # a failed recall restarts
    assert failed.ease < s.ease


def test_sm2_ease_is_floored():
    s = MasteryState("lora", 0.5, NOW, ease=1.3)
    assert sm2(s, 0, NOW).ease == 1.3


# ---------- compute_update


def test_asking_for_an_explanation_is_weak_evidence_of_a_gap():
    changes = _update(_snapshot(), "chat_turn", {"intent": "learn", "concepts": ["lora"]})
    [row] = changes.mastery
    prior = PRIOR["intermediate"]
    assert row.mastery < prior and row.mastery > prior - 0.1  # moved a little, not a lot
    assert row.evidence == 1
    assert row.next_review_at == NOW + timedelta(days=1)  # first exposure schedules a review
    assert changes.audit[0].field == "mastery:lora" and changes.audit[0].source == "inferred"


def test_quiz_is_strong_evidence_and_drives_the_schedule():
    snap = _snapshot([MasteryState("lora", 0.5, NOW)])
    changes = _update(snap, "quiz", {"concept": "lora", "correct": True})
    [row] = changes.mastery
    assert row.mastery == pytest.approx(0.65)  # alpha 0.3 * weight 1.0 * (1 - 0.5)
    assert row.reviews == 1


def test_confused_feedback_lowers_mastery_and_resets_the_review():
    snap = _snapshot([MasteryState("lora", 0.8, NOW, reviews=3, interval_days=15)])
    [row] = _update(snap, "feedback", {"signal": "confused", "concepts": ["lora"]}).mastery
    assert row.mastery < 0.8 and row.interval_days == 1 and row.reviews == 0


def test_answer_ratings_do_not_touch_mastery():
    changes = _update(_snapshot(), "feedback", {"signal": "helpful", "concepts": ["lora"]})
    assert changes.mastery == [] and changes.profile_patch == {}


def test_an_explicitly_locked_concept_is_not_inferred_over():
    snap = _snapshot([MasteryState("lora", 0.9, NOW, locked=True)])
    changes = _update(snap, "quiz", {"concept": "lora", "correct": False})
    assert changes.mastery == []


def test_decay_is_applied_before_new_evidence():
    old = MasteryState("lora", 0.95, NOW - timedelta(days=60))  # two half-lives ago
    [row] = _update(_snapshot([old]), "chat_turn", {"intent": "chat", "concepts": ["lora"]}).mastery
    assert row.mastery < 0.6  # decayed toward the 0.45 prior first, then nudged


def _strong(concept):
    return MasteryState(concept, 0.9, NOW, evidence=5)


def test_expertise_is_inferred_from_enough_evidence_and_audited():
    snap = _snapshot([_strong("lora"), _strong("rlhf"), _strong("dpo")])
    changes = _update(snap, "chat_turn", {"intent": "critique", "concepts": ["lora"]})
    assert changes.profile_patch["expertise_level"] == "advanced"
    assert any(a.field == "expertise_level" and a.new_value == "advanced" for a in changes.audit)


def test_expertise_is_not_inferred_from_thin_evidence():
    snap = _snapshot([_strong("lora")])
    changes = _update(snap, "chat_turn", {"intent": "critique", "concepts": ["lora"]})
    assert "expertise_level" not in changes.profile_patch


def test_explicit_expertise_pauses_inference():
    snap = _snapshot(
        [_strong("lora"), _strong("rlhf"), _strong("dpo")],
        profile={"expertise_level": "beginner"},
        locked=["expertise_level"],
    )
    changes = _update(snap, "chat_turn", {"intent": "critique", "concepts": ["lora"]})
    assert "expertise_level" not in changes.profile_patch


def test_interests_accumulate_and_decay():
    snap = _snapshot(profile={"inferred_interests": {"gan": 4.0}})
    snap.profile = replace(snap.profile, updated_at=NOW - timedelta(days=30))
    patch = _update(snap, "chat_turn", {"intent": "learn", "concepts": ["lora"]}).profile_patch
    assert patch["inferred_interests"] == {"gan": 2.0, "lora": 1.0}


# ---------- explicit edits


def test_explicit_changes_are_applied_locked_and_audited():
    state = ProfileState(dict(DEFAULT_PROFILE), [], 0, NOW)
    patch, locked, audit = explicit_profile_changes(
        state, {"expertise_level": "advanced", "bogus": 1}, []
    )
    assert patch == {"expertise_level": "advanced"}  # unknown fields ignored
    assert locked == ["expertise_level"]
    assert [(a.field, a.old_value, a.new_value, a.source) for a in audit] == [
        ("expertise_level", "intermediate", "advanced", "explicit")
    ]


def test_unlock_hands_the_field_back_to_inference():
    state = ProfileState(dict(DEFAULT_PROFILE), ["expertise_level"], 3, NOW)
    patch, locked, audit = explicit_profile_changes(state, {}, ["expertise_level"])
    assert patch == {} and locked == []
    assert audit[0].field == "lock:expertise_level"
