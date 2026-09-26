"""User accounts. Emails are stored lower-cased; uniqueness is enforced by the database."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from src.core.exceptions import ConflictError
from src.storage.database import User, get_async_session


@dataclass
class UserRecord:
    id: int
    email: str
    password_hash: str
    display_name: str | None
    is_admin: bool
    created_at: datetime


class UserStore(Protocol):
    async def create(
        self, email: str, password_hash: str, display_name: str | None, is_admin: bool
    ) -> UserRecord: ...

    async def by_email(self, email: str) -> UserRecord | None: ...

    async def by_id(self, user_id: int) -> UserRecord | None: ...

    async def delete(self, user_id: int) -> None: ...


def _record(row: User) -> UserRecord:
    return UserRecord(
        row.id, row.email, row.password_hash, row.display_name, row.is_admin, row.created_at
    )


class PgUserStore:
    async def create(self, email, password_hash, display_name, is_admin) -> UserRecord:
        async with get_async_session() as session:
            # ON CONFLICT instead of check-then-insert: two concurrent sign-ups cannot both win.
            result = await session.execute(
                insert(User)
                .values(
                    email=email.lower(),
                    password_hash=password_hash,
                    display_name=display_name,
                    is_admin=is_admin,
                )
                .on_conflict_do_nothing(index_elements=["email"])
                .returning(User)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise ConflictError("An account with this email already exists", code="EMAIL_TAKEN")
            await session.commit()
            return _record(row)

    async def by_email(self, email) -> UserRecord | None:
        async with get_async_session() as session:
            row = (
                await session.execute(select(User).where(User.email == email.lower()))
            ).scalar_one_or_none()
            return _record(row) if row else None

    async def by_id(self, user_id) -> UserRecord | None:
        async with get_async_session() as session:
            row = await session.get(User, user_id)
            return _record(row) if row else None

    async def delete(self, user_id) -> None:
        """Everything the user owns goes with the row (ON DELETE CASCADE): the GDPR erasure."""
        async with get_async_session() as session:
            await session.execute(delete(User).where(User.id == user_id))
            await session.commit()


class InMemoryUserStore:
    def __init__(self) -> None:
        self.users: dict[int, UserRecord] = {}
        self._next_id = 1

    async def create(self, email, password_hash, display_name, is_admin) -> UserRecord:
        if await self.by_email(email):
            raise ConflictError("An account with this email already exists", code="EMAIL_TAKEN")
        record = UserRecord(
            self._next_id,
            email.lower(),
            password_hash,
            display_name,
            is_admin,
            datetime.now(UTC),
        )
        self.users[record.id] = record
        self._next_id += 1
        return record

    async def by_email(self, email) -> UserRecord | None:
        return next((u for u in self.users.values() if u.email == email.lower()), None)

    async def by_id(self, user_id) -> UserRecord | None:
        return self.users.get(user_id)

    async def delete(self, user_id) -> None:
        self.users.pop(user_id, None)
