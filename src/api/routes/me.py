"""The signed-in user's own data: profile, mastery, reviews, audit, export and erasure (GDPR)."""

from datetime import UTC, datetime
from functools import partial
from typing import Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, Field

from src.agents import tutor as tutor_mod
from src.api.deps import current_user, services
from src.api.services import Services
from src.auth.security import verify_password
from src.auth.users import UserRecord
from src.core.config import settings
from src.core.exceptions import AuthError, ValidationError
from src.memory import concepts as vocab
from src.memory.model import (
    LOCKABLE,
    MasteryState,
    effective_mastery,
    explicit_profile_changes,
    prior_for,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Me"])

Lockable = Literal["expertise_level", "math_depth", "goals", "interests"]


class ProfileOut(BaseModel):
    user_id: int
    email: str
    display_name: str | None
    is_admin: bool
    profile: dict
    locked_fields: list[str]


class ProfileUpdate(BaseModel):
    """Explicit values: applied, audited, and locked against inference until unlocked."""

    expertise_level: Literal["beginner", "intermediate", "advanced"] | None = None
    math_depth: Literal["light", "standard", "rigorous"] | None = None
    goals: list[str] | None = Field(default=None, max_length=10)
    interests: list[str] | None = Field(default=None, max_length=20)
    unlock: list[Lockable] = Field(default_factory=list)


class MasteryOut(BaseModel):
    concept: str
    label: str
    mastery: float  # effective (decayed) value now
    stored: float  # value at the last update
    evidence: int
    locked: bool
    next_review_at: datetime | None


class MasteryUpdate(BaseModel):
    mastery: float | None = Field(..., ge=0, le=1, description="null unlocks (back to inference)")


class DeleteAccount(BaseModel):
    password: str  # re-authentication: a stolen token alone cannot erase an account


async def _profile_out(svc: Services, user: UserRecord) -> ProfileOut:
    state = await svc.profiles.get_profile(user.id)
    return ProfileOut(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_admin=user.is_admin,
        profile=state.profile,
        locked_fields=state.locked,
    )


def _mastery_out(state: MasteryState, prior: float, now: datetime) -> MasteryOut:
    return MasteryOut(
        concept=state.concept,
        label=vocab.label(state.concept),
        mastery=round(
            effective_mastery(state, prior, now, settings.MEMORY_DECAY_HALF_LIFE_DAYS), 3
        ),
        stored=round(state.mastery, 3),
        evidence=state.evidence,
        locked=state.locked,
        next_review_at=state.next_review_at,
    )


@router.get("/me", response_model=ProfileOut)
async def me(user: UserRecord = Depends(current_user), svc: Services = Depends(services)):
    return await _profile_out(svc, user)


@router.patch("/me/profile", response_model=ProfileOut)
async def update_profile(
    body: ProfileUpdate,
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    updates = body.model_dump(exclude_none=True, exclude={"unlock"})
    if not updates and not body.unlock:
        raise ValidationError("Nothing to update", code="EMPTY_UPDATE")
    unlock = [f for f in body.unlock if f in LOCKABLE]
    await svc.profiles.update_profile_explicit(
        user.id, partial(explicit_profile_changes, updates=updates, unlock=unlock)
    )
    logger.info(
        "Profile updated explicitly", user_id=user.id, fields=sorted(updates), unlock=unlock
    )
    return await _profile_out(svc, user)


@router.get("/me/mastery", response_model=list[MasteryOut])
async def mastery(user: UserRecord = Depends(current_user), svc: Services = Depends(services)):
    now = datetime.now(UTC)
    prior = prior_for((await svc.profiles.get_profile(user.id)).profile)
    rows = [_mastery_out(s, prior, now) for s in await svc.profiles.get_mastery(user.id)]
    return sorted(rows, key=lambda m: m.mastery)


@router.put("/me/mastery/{concept}", response_model=MasteryOut)
async def set_mastery(
    concept: str,
    body: MasteryUpdate,
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    if not vocab.is_known(concept):
        raise ValidationError(f"Unknown concept '{concept}'", code="UNKNOWN_CONCEPT")
    now = datetime.now(UTC)
    state = await svc.profiles.set_mastery_explicit(user.id, concept, body.mastery, now)
    prior = prior_for((await svc.profiles.get_profile(user.id)).profile)
    return _mastery_out(state, prior, now)


@router.get("/me/reviews", response_model=list[MasteryOut])
async def due_reviews(
    limit: int = Query(default=10, ge=1, le=50),
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    """Concepts whose spaced-repetition review is due (the tutor's quiz queue in Phase 5)."""
    now = datetime.now(UTC)
    prior = prior_for((await svc.profiles.get_profile(user.id)).profile)
    return [
        _mastery_out(s, prior, now) for s in await svc.profiles.due_reviews(user.id, now, limit)
    ]


@router.get("/me/audit")
async def audit(
    limit: int = Query(default=50, ge=1, le=500),
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    """Every change to the profile and to mastery, with its source (explicit or inferred)."""
    return await svc.profiles.audit(user.id, limit)


@router.get("/me/export")
async def export(user: UserRecord = Depends(current_user), svc: Services = Depends(services)):
    """GDPR Art. 15/20: everything stored about the user, machine-readable."""
    data = await svc.profiles.export(user.id)
    return {
        "exported_at": datetime.now(UTC).isoformat(),
        "account": {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "created_at": user.created_at.isoformat(),
        },
        **data,
        "conversations": await svc.conversations.export(user.id),
    }


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    body: DeleteAccount,
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    """GDPR Art. 17: erase the account and everything it owns."""
    if not verify_password(body.password, user.password_hash):
        raise AuthError("Password is incorrect", code="INVALID_CREDENTIALS")
    await svc.conversations.delete_user(user.id)
    await svc.profiles.delete_user(user.id)
    await svc.users.delete(user.id)  # Postgres: ON DELETE CASCADE removes the rest
    logger.info("Account deleted", user_id=user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/conversations")
async def list_conversations(
    limit: int = Query(default=30, ge=1, le=100),
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    return await svc.conversations.list_conversations(user.id, limit)


@router.get("/me/papers")
async def studied_papers(
    limit: int = Query(default=200, ge=1, le=500),
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    """Papers your answers have cited, most recent first (the map marks them as studied)."""
    return await svc.conversations.cited_papers(user.id, limit)


@router.get("/conversations/{conversation_id}/tutor")
async def conversation_tutor(
    conversation_id: UUID,
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    """The tutoring session of a conversation (pending question, step, reading position)."""
    state = await svc.conversations.get_state(user.id, str(conversation_id))
    return {"tutor": tutor_mod.public_view(state)}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    user: UserRecord = Depends(current_user),
    svc: Services = Depends(services),
):
    return await svc.conversations.messages(user.id, str(conversation_id))
