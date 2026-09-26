"""Chat history persistence (ADR 0007).

The chat loop depends on the `ConversationStore` protocol, not on Postgres, so tests (and a
database outage) can use `InMemoryConversationStore`. History is best effort: the API answers even
when it cannot load or save history, because an answer without context beats no answer.
"""

from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from src.storage.database import Conversation, Message, get_async_session


class ConversationStore(Protocol):
    async def history(self, conversation_id: str, limit: int) -> list[dict]: ...

    async def append(
        self, conversation_id: str, messages: list[dict], title: str | None = None
    ) -> None: ...


class PgConversationStore:
    async def history(self, conversation_id: str, limit: int) -> list[dict]:
        """The last `limit` messages, oldest first."""
        async with get_async_session() as session:
            rows = (
                await session.execute(
                    select(Message.role, Message.content)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.id.desc())
                    .limit(limit)
                )
            ).all()
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    async def append(
        self, conversation_id: str, messages: list[dict], title: str | None = None
    ) -> None:
        """Creates the conversation on first use; one transaction for the whole turn."""
        async with get_async_session() as session:
            await session.execute(
                insert(Conversation)
                .values(id=conversation_id, title=title)
                .on_conflict_do_nothing(index_elements=["id"])
            )
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


class InMemoryConversationStore:
    def __init__(self) -> None:
        self.data: dict[str, list[dict]] = {}

    async def history(self, conversation_id: str, limit: int) -> list[dict]:
        rows = self.data.get(conversation_id, [])[-limit:]
        return [{"role": m["role"], "content": m["content"]} for m in rows]

    async def append(
        self, conversation_id: str, messages: list[dict], title: str | None = None
    ) -> None:
        self.data.setdefault(conversation_id, []).extend(messages)
