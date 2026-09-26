"""Chat history persistence (ADR 0007), scoped to the owning user (ADR 0008).

Ownership is checked on every access: a conversation id is a client-supplied value, so without the
check anyone who guessed or saw an id could read another user's history. A foreign id answers
404 (not 403), so the API does not even confirm the conversation exists.

The chat loop depends on the `ConversationStore` protocol, not on Postgres, so tests use
`InMemoryConversationStore`. History is best effort for availability, but ownership is not:
NotFoundError always propagates.
"""

from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from src.core.exceptions import NotFoundError
from src.storage.database import Conversation, Message, get_async_session

NOT_FOUND = "Conversation not found"


class ConversationStore(Protocol):
    async def history(self, user_id: int, conversation_id: str, limit: int) -> list[dict]: ...

    async def append(
        self, user_id: int, conversation_id: str, messages: list[dict], title: str | None = None
    ) -> None: ...

    async def list_conversations(self, user_id: int, limit: int) -> list[dict]: ...

    async def messages(self, user_id: int, conversation_id: str) -> list[dict]: ...

    async def count_user_messages_since(self, user_id: int, since: datetime) -> int: ...

    async def export(self, user_id: int) -> list[dict]: ...

    async def delete_user(self, user_id: int) -> None: ...


class PgConversationStore:
    async def _owner(self, session, conversation_id: str) -> int | None | bool:
        """The owner's id; False if the conversation does not exist."""
        row = (
            await session.execute(
                select(Conversation.user_id).where(Conversation.id == conversation_id)
            )
        ).one_or_none()
        return False if row is None else row.user_id

    async def history(self, user_id, conversation_id, limit) -> list[dict]:
        """The last `limit` messages, oldest first. A new id has no history (not an error)."""
        async with get_async_session() as session:
            owner = await self._owner(session, conversation_id)
            if owner is False:
                return []
            if owner != user_id:
                raise NotFoundError(NOT_FOUND, code="CONVERSATION_NOT_FOUND")
            rows = (
                await session.execute(
                    select(Message.role, Message.content)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.id.desc())
                    .limit(limit)
                )
            ).all()
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    async def append(self, user_id, conversation_id, messages, title=None) -> None:
        """Creates the conversation on first use; one transaction for the whole turn."""
        async with get_async_session() as session:
            await session.execute(
                insert(Conversation)
                .values(id=conversation_id, user_id=user_id, title=title)
                .on_conflict_do_nothing(index_elements=["id"])
            )
            if await self._owner(session, conversation_id) != user_id:
                await session.rollback()
                raise NotFoundError(NOT_FOUND, code="CONVERSATION_NOT_FOUND")
            await session.execute(
                insert(Message).values(
                    [{"conversation_id": conversation_id, **m} for m in messages]
                )
            )
            await session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(updated_at=datetime.now(UTC))
            )
            await session.commit()

    async def list_conversations(self, user_id, limit) -> list[dict]:
        async with get_async_session() as session:
            rows = (
                await session.execute(
                    select(Conversation)
                    .where(Conversation.user_id == user_id)
                    .order_by(Conversation.updated_at.desc())
                    .limit(limit)
                )
            ).scalars()
            return [
                {"id": r.id, "title": r.title, "updated_at": r.updated_at.isoformat()} for r in rows
            ]

    async def messages(self, user_id, conversation_id) -> list[dict]:
        async with get_async_session() as session:
            if await self._owner(session, conversation_id) != user_id:
                raise NotFoundError(NOT_FOUND, code="CONVERSATION_NOT_FOUND")
            rows = (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.id)
                )
            ).scalars()
            return [_message(r) for r in rows]

    async def count_user_messages_since(self, user_id, since) -> int:
        async with get_async_session() as session:
            return (
                await session.execute(
                    select(func.count())
                    .select_from(Message)
                    .join(Conversation, Conversation.id == Message.conversation_id)
                    .where(
                        Conversation.user_id == user_id,
                        Message.role == "user",
                        Message.created_at >= since,
                    )
                )
            ).scalar_one()

    async def export(self, user_id) -> list[dict]:
        conversations = await self.list_conversations(user_id, limit=100_000)
        for c in conversations:
            c["messages"] = await self.messages(user_id, c["id"])
        return conversations

    async def delete_user(self, user_id) -> None:
        """Nothing to do: conversations reference users(id) ON DELETE CASCADE."""


def _message(r: Message) -> dict:
    return {
        "role": r.role,
        "content": r.content,
        "intent": r.intent,
        "citations": r.citations,
        "provider": r.provider,
        "at": r.created_at.isoformat(),
    }


class InMemoryConversationStore:
    def __init__(self) -> None:
        self.data: dict[str, list[dict]] = {}
        self.owners: dict[str, int] = {}

    def _check(self, user_id: int, conversation_id: str) -> bool:
        owner = self.owners.get(conversation_id)
        if owner is not None and owner != user_id:
            raise NotFoundError(NOT_FOUND, code="CONVERSATION_NOT_FOUND")
        return owner is not None

    async def history(self, user_id, conversation_id, limit):
        if not self._check(user_id, conversation_id):
            return []
        rows = self.data.get(conversation_id, [])[-limit:]
        return [{"role": m["role"], "content": m["content"]} for m in rows]

    async def append(self, user_id, conversation_id, messages, title=None):
        self._check(user_id, conversation_id)
        self.owners[conversation_id] = user_id
        now = datetime.now(UTC)
        self.data.setdefault(conversation_id, []).extend({**m, "at": now} for m in messages)

    async def list_conversations(self, user_id, limit):
        ids = [cid for cid, owner in self.owners.items() if owner == user_id]
        return [{"id": cid, "title": None} for cid in reversed(ids)][:limit]

    async def messages(self, user_id, conversation_id):
        if not self._check(user_id, conversation_id):
            raise NotFoundError(NOT_FOUND, code="CONVERSATION_NOT_FOUND")
        return list(self.data[conversation_id])

    async def count_user_messages_since(self, user_id, since):
        return sum(
            1
            for cid, owner in self.owners.items()
            if owner == user_id
            for m in self.data.get(cid, [])
            if m["role"] == "user" and m["at"] >= since
        )

    async def export(self, user_id):
        return [
            {"id": c["id"], "messages": await self.messages(user_id, c["id"])}
            for c in await self.list_conversations(user_id, 100_000)
        ]

    async def delete_user(self, user_id):
        for cid in [c for c, o in self.owners.items() if o == user_id]:
            self.owners.pop(cid)
            self.data.pop(cid, None)
