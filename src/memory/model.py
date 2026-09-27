"""The learner model: pure functions, no I/O (ADR 0008).

Every number that changes a user's profile is computed here from (snapshot, event, now), so the
whole model is unit-testable and the stores only have to apply the result atomically.

Mastery m in [0, 1] per concept:
  observation   each signal is an observation o with a weight w (how strong the evidence is)
  update        m <- clamp(m + alpha * w * (o - m))                       (weighted EMA)
  decay         m_eff(t) = prior + (m - prior) * 2^(-dt / half_life)      (forgetting)
  prior         what we assume before any evidence: from the user's expertise level
  lock          an explicit value set by the user: never decayed or inferred over

Review schedule: SM-2 (the Anki/SuperMemo algorithm) driven by a 0-5 recall quality.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

EXPERTISE_LEVELS = ("beginner", "intermediate", "advanced")
MATH_DEPTHS = ("light", "standard", "rigorous")
PRIOR = {"beginner": 0.2, "intermediate": 0.45, "advanced": 0.7}
DEFAULT_PROFILE: dict = {
    "expertise_level": "intermediate",
    "math_depth": "standard",
    "goals": [],
    "interests": [],
    "inferred_interests": {},  # concept -> weight; inference only, never locked
}
LOCKABLE = ("expertise_level", "math_depth", "goals", "interests")

# signal -> (observation, weight). Asking to have something explained is weak evidence of a gap;
# asking for a critique or comparison is weak evidence of fluency; explicit feedback is stronger;
# a quiz answer is the strongest evidence we get.
SIGNALS: dict[str, tuple[float, float]] = {
    "asked_learn": (0.25, 0.3),
    "asked_critique": (0.7, 0.3),
    "asked_synthesize": (0.7, 0.3),
    "asked_recommend": (0.5, 0.1),
    "asked_chat": (0.5, 0.1),
    "understood": (0.8, 0.6),
    "confused": (0.1, 0.6),
    "quiz_correct": (1.0, 1.0),
    "quiz_partial": (0.5, 0.8),
    "quiz_incorrect": (0.0, 1.0),
}
# SM-2 recall quality (0-5) for signals that are a genuine recall test; plain questions are not.
REVIEW_QUALITY = {
    "quiz_correct": 5,
    "understood": 4,
    "quiz_partial": 3,
    "confused": 1,
    "quiz_incorrect": 0,
}

# Expertise inference: the mean mastery over concepts with enough evidence.
MIN_EVIDENCE = 3
MIN_CONCEPTS = 3
EXPERTISE_THRESHOLDS = ((0.35, "beginner"), (0.65, "intermediate"), (1.01, "advanced"))
INTEREST_TOP_N = 20


def clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


@dataclass
class MasteryState:
    concept: str
    mastery: float
    updated_at: datetime
    evidence: int = 0
    locked: bool = False
    ease: float = 2.5
    interval_days: float = 0.0
    reviews: int = 0
    next_review_at: datetime | None = None


@dataclass
class ProfileState:
    profile: dict
    locked: list[str]
    version: int
    updated_at: datetime


@dataclass
class Snapshot:
    profile: ProfileState
    mastery: dict[str, MasteryState]


@dataclass
class AuditEntry:
    field: str
    old_value: object
    new_value: object
    source: str  # "explicit" | "inferred"


@dataclass
class Changes:
    mastery: list[MasteryState] = field(default_factory=list)
    profile_patch: dict = field(default_factory=dict)
    audit: list[AuditEntry] = field(default_factory=list)


# ------------------------------------------------------------------ primitives


def prior_for(profile: dict) -> float:
    return PRIOR.get(profile.get("expertise_level", "intermediate"), PRIOR["intermediate"])


def decayed(value: float, prior: float, since: datetime, now: datetime, half_life: float) -> float:
    days = max(0.0, (now - since).total_seconds() / 86400)
    return clamp(prior + (value - prior) * 0.5 ** (days / half_life))


def effective_mastery(state: MasteryState, prior: float, now: datetime, half_life: float) -> float:
    if state.locked:
        return state.mastery
    return decayed(state.mastery, prior, state.updated_at, now, half_life)


def ema(current: float, observation: float, weight: float, alpha: float) -> float:
    return clamp(current + alpha * weight * (observation - current))


def sm2(state: MasteryState, quality: int, now: datetime) -> MasteryState:
    """One SM-2 step. Failed recall (q < 3) restarts the ladder at 1 day; each success lengthens
    the interval (1, 6, then x ease). Ease drops on hard recalls, floored at 1.3."""
    if quality < 3:
        reviews, interval = 0, 1.0
    else:
        reviews = state.reviews + 1
        interval = (
            1.0 if reviews == 1 else 6.0 if reviews == 2 else state.interval_days * state.ease
        )
    ease = max(1.3, state.ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return replace(
        state,
        ease=round(ease, 3),
        interval_days=round(interval, 2),
        reviews=reviews,
        next_review_at=now + timedelta(days=interval),
    )


def infer_expertise(states: list[MasteryState], prior: float, now: datetime, half_life: float):
    eligible = [
        effective_mastery(s, prior, now, half_life) for s in states if s.evidence >= MIN_EVIDENCE
    ]
    if len(eligible) < MIN_CONCEPTS:
        return None  # not enough evidence to say anything
    mean = sum(eligible) / len(eligible)
    return next(level for bound, level in EXPERTISE_THRESHOLDS if mean < bound)


def signals_for(kind: str, payload: dict) -> list[tuple[str, str]]:
    """(concept, signal) pairs carried by one event."""
    if kind == "chat_turn":
        signal = f"asked_{payload.get('intent', 'chat')}"
        return [(c, signal) for c in payload.get("concepts", []) if signal in SIGNALS]
    if kind == "feedback" and payload.get("signal") in ("understood", "confused"):
        return [(c, payload["signal"]) for c in payload.get("concepts", [])]
    if kind == "quiz" and payload.get("concept"):
        result = payload.get("result") or ("correct" if payload.get("correct") else "incorrect")
        signal = f"quiz_{result}"
        return [(payload["concept"], signal)] if signal in SIGNALS else []
    return []  # e.g. helpful/not_helpful rate the answer, not the learner


# ------------------------------------------------------------------ the update


def compute_update(
    snapshot: Snapshot, kind: str, payload: dict, now: datetime, alpha: float, half_life: float
) -> Changes:
    changes = Changes()
    profile = snapshot.profile.profile
    prior = prior_for(profile)
    states = dict(snapshot.mastery)
    pairs = signals_for(kind, payload)

    for concept, signal in pairs:
        state = states.get(concept) or MasteryState(concept, prior, now)
        if state.locked:
            continue  # explicit override: inference paused for this concept
        before = effective_mastery(state, prior, now, half_life)
        observation, weight = SIGNALS[signal]
        after = ema(before, observation, weight, alpha)
        new = replace(state, mastery=round(after, 4), updated_at=now, evidence=state.evidence + 1)
        if signal in REVIEW_QUALITY:
            new = sm2(new, REVIEW_QUALITY[signal], now)
        elif new.next_review_at is None:
            new = replace(new, next_review_at=now + timedelta(days=1))  # first exposure
        states[concept] = new
        changes.mastery.append(new)
        if round(after, 3) != round(before, 3) or concept not in snapshot.mastery:
            changes.audit.append(
                AuditEntry(f"mastery:{concept}", round(before, 3), round(after, 3), "inferred")
            )

    touched = [c for c, _ in pairs]
    if touched:
        changes.profile_patch["inferred_interests"] = _interests(
            profile.get("inferred_interests", {}),
            touched,
            snapshot.profile.updated_at,
            now,
            half_life,
        )

    if "expertise_level" not in snapshot.profile.locked:
        inferred = infer_expertise(list(states.values()), prior, now, half_life)
        current = profile.get("expertise_level")
        if inferred and inferred != current:
            changes.profile_patch["expertise_level"] = inferred
            changes.audit.append(AuditEntry("expertise_level", current, inferred, "inferred"))
    return changes


def _interests(
    current: dict[str, float],
    touched: list[str],
    since: datetime,
    now: datetime,
    half_life: float,
) -> dict[str, float]:
    days = max(0.0, (now - since).total_seconds() / 86400)
    factor = 0.5 ** (days / half_life)
    weights = {c: w * factor for c, w in current.items()}
    for concept in touched:
        weights[concept] = weights.get(concept, 0.0) + 1.0
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:INTEREST_TOP_N]
    return {c: round(w, 3) for c, w in top if w >= 0.01}


# ------------------------------------------------------------------ explicit edits


def explicit_profile_changes(
    state: ProfileState, updates: dict, unlock: Sequence[str]
) -> tuple[dict, list[str], list[AuditEntry]]:
    """User-set fields: applied, locked against inference, and audited. `unlock` hands a field
    back to inference (its current value stays until the next inferred change)."""
    patch, audit = {}, []
    locked = [f for f in state.locked if f not in unlock]
    for name, value in updates.items():
        if name not in LOCKABLE:
            continue
        old = state.profile.get(name)
        if old != value:
            patch[name] = value
            audit.append(AuditEntry(name, old, value, "explicit"))
        if name not in locked:
            locked.append(name)
    for name in unlock:
        if name in state.locked:
            audit.append(AuditEntry(f"lock:{name}", True, False, "explicit"))
    return patch, locked, audit
