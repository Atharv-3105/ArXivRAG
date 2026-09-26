"""Persistence for profiles, mastery, events and the audit trail (ADR 0008).

Consistency rules, identical in both implementations:
  * apply_event is all-or-nothing: event row + mastery rows + profile patch + audit rows commit
    together, so a retried event can never be half-applied.
  * The event id is the idempotency key: inserting it is the first statement; a conflict means
    "already applied" and the whole event is skipped.
  * Profile writes are optimistic: `UPDATE ... WHERE version = :seen`. Zero rows updated means
    someone else changed the profile in between -> ConcurrentUpdate, the caller retries on a fresh
    snapshot. No row locks are held while computing.
  * A mastery upsert never overwrites a row the user locked in the meantime (`WHERE NOT locked`).
"""

import copy
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import bindparam, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert

from src.memory.model import (
    DEFAULT_PROFILE,
    AuditEntry,
    Changes,
    MasteryState,
    ProfileState,
    Snapshot,
)
from src.storage.database import (
    ConceptMastery,
    ProfileAudit,
    UserEvent,
    UserProfile,
    get_async_session,
)

ComputeFn = Callable[[Snapshot], Changes]


class ConcurrentUpdate(Exception):
    """The profile version changed between read and write; retry with a fresh snapshot."""


class ProfileStore(Protocol):
    async def get_profile(self, user_id: int) -> ProfileState: ...

    async def get_mastery(
        self, user_id: int, concepts: list[str] | None = None
    ) -> list[MasteryState]: ...

    async def apply_event(
        self, user_id: int, event_id: str, kind: str, payload: dict, compute: ComputeFn
    ) -> bool: ...

    async def update_profile_explicit(
        self,
        user_id: int,
        compute: Callable[[ProfileState], tuple[dict, list[str], list[AuditEntry]]],
    ) -> ProfileState: ...

    async def set_mastery_explicit(
        self, user_id: int, concept: str, value: float | None, now: datetime
    ) -> MasteryState: ...

    async def due_reviews(self, user_id: int, now: datetime, limit: int) -> list[MasteryState]: ...

    async def audit(self, user_id: int, limit: int) -> list[dict]: ...

    async def export(self, user_id: int) -> dict: ...

    async def delete_user(self, user_id: int) -> None: ...


def _mastery_row(user_id: int, s: MasteryState) -> dict:
    return {"user_id": user_id, **asdict(s)}


def _state(row: ConceptMastery) -> MasteryState:
    return MasteryState(
        concept=row.concept,
        mastery=row.mastery,
        updated_at=row.updated_at,
        evidence=row.evidence,
        locked=row.locked,
        ease=row.ease,
        interval_days=row.interval_days,
        reviews=row.reviews,
        next_review_at=row.next_review_at,
    )


def _jsonable(value):
    return (
        value
        if isinstance(value, dict | list | str | int | float | bool) or value is None
        else str(value)
    )


# ================================================================== Postgres


class PgProfileStore:
    async def _profile_row(self, session, user_id: int) -> UserProfile:
        await session.execute(
            insert(UserProfile)
            .values(user_id=user_id, profile=DEFAULT_PROFILE, locked=[])
            .on_conflict_do_nothing(index_elements=["user_id"])
        )
        row = (
            await session.execute(select(UserProfile).where(UserProfile.user_id == user_id))
        ).scalar_one()
        return row

    @staticmethod
    def _profile_state(row: UserProfile) -> ProfileState:
        return ProfileState(
            profile={**DEFAULT_PROFILE, **(row.profile or {})},
            locked=list(row.locked or []),
            version=row.version,
            updated_at=row.updated_at,
        )

    async def get_profile(self, user_id: int) -> ProfileState:
        async with get_async_session() as session:
            row = await self._profile_row(session, user_id)
            await session.commit()
            return self._profile_state(row)

    async def get_mastery(self, user_id: int, concepts: list[str] | None = None):
        async with get_async_session() as session:
            query = select(ConceptMastery).where(ConceptMastery.user_id == user_id)
            if concepts is not None:
                query = query.where(ConceptMastery.concept.in_(concepts))
            return [_state(r) for r in (await session.execute(query)).scalars()]

    async def _patch_profile(self, session, user_id: int, seen: int, patch: dict, locked=None):
        """Atomic JSONB merge (`profile || patch`) guarded by the version we computed from."""
        values: dict = {
            "profile": UserProfile.profile.op("||")(bindparam("patch", patch, type_=JSONB)),
            "version": UserProfile.version + 1,
            "updated_at": datetime.now(UTC),
        }
        if locked is not None:
            values["locked"] = locked
        result = await session.execute(
            update(UserProfile)
            .where(UserProfile.user_id == user_id, UserProfile.version == seen)
            .values(**values)
        )
        if result.rowcount != 1:
            raise ConcurrentUpdate(f"profile of user {user_id} changed (version {seen})")

    async def _write_audit(self, session, user_id: int, entries: list[AuditEntry], event_id=None):
        if entries:
            await session.execute(
                insert(ProfileAudit).values(
                    [
                        {
                            "user_id": user_id,
                            "field": e.field,
                            "old_value": _jsonable(e.old_value),
                            "new_value": _jsonable(e.new_value),
                            "source": e.source,
                            "event_id": event_id,
                        }
                        for e in entries
                    ]
                )
            )

    async def apply_event(self, user_id, event_id, kind, payload, compute) -> bool:
        async with get_async_session() as session:
            inserted = await session.execute(
                insert(UserEvent)
                .values(event_id=event_id, user_id=user_id, kind=kind, payload=payload)
                .on_conflict_do_nothing(index_elements=["event_id"])
                .returning(UserEvent.event_id)
            )
            if inserted.scalar_one_or_none() is None:
                return False  # already applied: idempotent no-op

            profile_row = await self._profile_row(session, user_id)
            mastery_rows = (
                await session.execute(
                    select(ConceptMastery).where(ConceptMastery.user_id == user_id)
                )
            ).scalars()
            snapshot = Snapshot(
                self._profile_state(profile_row), {r.concept: _state(r) for r in mastery_rows}
            )
            changes = compute(snapshot)

            for state in changes.mastery:
                row = _mastery_row(user_id, state)
                stmt = insert(ConceptMastery).values(row)
                await session.execute(
                    stmt.on_conflict_do_update(
                        index_elements=["user_id", "concept"],
                        set_={k: stmt.excluded[k] for k in row if k not in ("user_id", "concept")},
                        where=ConceptMastery.locked.is_(False),
                    )
                )
            if changes.profile_patch:
                await self._patch_profile(
                    session, user_id, snapshot.profile.version, changes.profile_patch
                )
            await self._write_audit(session, user_id, changes.audit, event_id)
            await session.commit()
            return True

    async def update_profile_explicit(self, user_id, compute) -> ProfileState:
        async with get_async_session() as session:
            state = self._profile_state(await self._profile_row(session, user_id))
            patch, locked, audit = compute(state)
            await self._patch_profile(session, user_id, state.version, patch, locked)
            await self._write_audit(session, user_id, audit)
            await session.commit()
        return await self.get_profile(user_id)

    async def set_mastery_explicit(self, user_id, concept, value, now) -> MasteryState:
        async with get_async_session() as session:
            existing = (
                await session.execute(
                    select(ConceptMastery).where(
                        ConceptMastery.user_id == user_id, ConceptMastery.concept == concept
                    )
                )
            ).scalar_one_or_none()
            old = _state(existing) if existing else None
            state = replace(
                old or MasteryState(concept, value if value is not None else 0.5, now),
                mastery=value if value is not None else (old.mastery if old else 0.5),
                locked=value is not None,
                updated_at=now,
            )
            row = _mastery_row(user_id, state)
            stmt = insert(ConceptMastery).values(row)
            await session.execute(
                stmt.on_conflict_do_update(
                    index_elements=["user_id", "concept"],
                    set_={k: stmt.excluded[k] for k in ("mastery", "locked", "updated_at")},
                )
            )
            await self._write_audit(
                session,
                user_id,
                [
                    AuditEntry(
                        f"mastery:{concept}",
                        old.mastery if old else None,
                        value if value is not None else "unlocked",
                        "explicit",
                    )
                ],
            )
            await session.commit()
            return state

    async def due_reviews(self, user_id, now, limit) -> list[MasteryState]:
        async with get_async_session() as session:
            rows = (
                await session.execute(
                    select(ConceptMastery)
                    .where(
                        ConceptMastery.user_id == user_id,
                        ConceptMastery.next_review_at <= now,
                    )
                    .order_by(ConceptMastery.next_review_at)
                    .limit(limit)
                )
            ).scalars()
            return [_state(r) for r in rows]

    async def audit(self, user_id, limit) -> list[dict]:
        async with get_async_session() as session:
            rows = (
                await session.execute(
                    select(ProfileAudit)
                    .where(ProfileAudit.user_id == user_id)
                    .order_by(ProfileAudit.id.desc())
                    .limit(limit)
                )
            ).scalars()
            return [
                {
                    "field": r.field,
                    "old_value": r.old_value,
                    "new_value": r.new_value,
                    "source": r.source,
                    "event_id": r.event_id,
                    "at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def export(self, user_id) -> dict:
        async with get_async_session() as session:
            events = (
                await session.execute(
                    select(UserEvent)
                    .where(UserEvent.user_id == user_id)
                    .order_by(UserEvent.created_at)
                )
            ).scalars()
            event_rows = [
                {
                    "event_id": e.event_id,
                    "kind": e.kind,
                    "payload": e.payload,
                    "at": e.created_at.isoformat(),
                }
                for e in events
            ]
        profile = await self.get_profile(user_id)
        return {
            "profile": profile.profile,
            "locked_fields": profile.locked,
            "mastery": [asdict(m) for m in await self.get_mastery(user_id)],
            "events": event_rows,
            "audit": await self.audit(user_id, limit=100_000),
        }

    async def delete_user(self, user_id) -> None:
        """Nothing to do: every table here references users(id) ON DELETE CASCADE."""


# ================================================================== in memory (tests, demos)


class InMemoryProfileStore:
    """Same semantics as PgProfileStore, including idempotency and the version check."""

    def __init__(self) -> None:
        self.profiles: dict[int, ProfileState] = {}
        self.mastery: dict[int, dict[str, MasteryState]] = {}
        self.events: dict[str, dict] = {}
        self.audit_log: list[dict] = []

    def _profile(self, user_id: int) -> ProfileState:
        if user_id not in self.profiles:
            self.profiles[user_id] = ProfileState(
                copy.deepcopy(DEFAULT_PROFILE), [], 0, datetime.now(UTC)
            )
        return self.profiles[user_id]

    async def get_profile(self, user_id):
        return copy.deepcopy(self._profile(user_id))

    async def get_mastery(self, user_id, concepts=None):
        rows = self.mastery.get(user_id, {})
        return [copy.deepcopy(s) for c, s in rows.items() if concepts is None or c in concepts]

    def _audit(self, user_id, entries, event_id=None):
        for e in entries:
            self.audit_log.append(
                {
                    "user_id": user_id,
                    **asdict(e),
                    "event_id": event_id,
                    "at": datetime.now(UTC).isoformat(),
                }
            )

    def _patch(self, user_id, seen, patch, locked=None):
        current = self._profile(user_id)
        if current.version != seen:
            raise ConcurrentUpdate(f"profile of user {user_id} changed (version {seen})")
        self.profiles[user_id] = ProfileState(
            {**current.profile, **patch},
            locked if locked is not None else current.locked,
            current.version + 1,
            datetime.now(UTC),
        )

    async def apply_event(self, user_id, event_id, kind, payload, compute) -> bool:
        if event_id in self.events:
            return False
        snapshot = Snapshot(
            copy.deepcopy(self._profile(user_id)), copy.deepcopy(self.mastery.get(user_id, {}))
        )
        changes = compute(snapshot)
        # Stage everything first so an exception leaves no partial write (like a rollback).
        if changes.profile_patch and self._profile(user_id).version != snapshot.profile.version:
            raise ConcurrentUpdate(f"profile of user {user_id} changed")
        rows = self.mastery.setdefault(user_id, {})
        for state in changes.mastery:
            if not (state.concept in rows and rows[state.concept].locked):
                rows[state.concept] = state
        if changes.profile_patch:
            self._patch(user_id, snapshot.profile.version, changes.profile_patch)
        self._audit(user_id, changes.audit, event_id)
        self.events[event_id] = {"user_id": user_id, "kind": kind, "payload": payload}
        return True

    async def update_profile_explicit(self, user_id, compute):
        state = copy.deepcopy(self._profile(user_id))
        patch, locked, audit = compute(state)
        self._patch(user_id, state.version, patch, locked)
        self._audit(user_id, audit)
        return await self.get_profile(user_id)

    async def set_mastery_explicit(self, user_id, concept, value, now):
        rows = self.mastery.setdefault(user_id, {})
        old = rows.get(concept)
        state = replace(
            old or MasteryState(concept, value if value is not None else 0.5, now),
            mastery=value if value is not None else (old.mastery if old else 0.5),
            locked=value is not None,
            updated_at=now,
        )
        rows[concept] = state
        self._audit(
            user_id,
            [
                AuditEntry(
                    f"mastery:{concept}",
                    old.mastery if old else None,
                    value if value is not None else "unlocked",
                    "explicit",
                )
            ],
        )
        return copy.deepcopy(state)

    async def due_reviews(self, user_id, now, limit):
        due: list[MasteryState] = [
            s
            for s in self.mastery.get(user_id, {}).values()
            if s.next_review_at is not None and s.next_review_at <= now
        ]
        return sorted(due, key=lambda s: s.next_review_at or now)[:limit]

    async def audit(self, user_id, limit):
        rows = [a for a in self.audit_log if a["user_id"] == user_id]
        return [{k: v for k, v in a.items() if k != "user_id"} for a in reversed(rows)][:limit]

    async def export(self, user_id):
        profile = await self.get_profile(user_id)
        return {
            "profile": profile.profile,
            "locked_fields": profile.locked,
            "mastery": [asdict(m) for m in await self.get_mastery(user_id)],
            "events": [
                {"event_id": k, **v} for k, v in self.events.items() if v["user_id"] == user_id
            ],
            "audit": await self.audit(user_id, limit=100_000),
        }

    async def delete_user(self, user_id: int) -> None:
        self.profiles.pop(user_id, None)
        self.mastery.pop(user_id, None)
        self.events = {k: v for k, v in self.events.items() if v["user_id"] != user_id}
        self.audit_log = [a for a in self.audit_log if a["user_id"] != user_id]
