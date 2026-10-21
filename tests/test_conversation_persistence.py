"""PgConversationStore.append writes every column of every message.

Regression: a single multi-row INSERT took its column set from the first row (the user message),
so the assistant row lost its citations, agent, provider, prompt version and checks. The
in-memory store cannot show that, so the statements the Postgres store issues are inspected.
"""

from contextlib import asynccontextmanager

from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import dml

from src.storage import conversations
from src.storage.conversations import PgConversationStore


class RecordingSession:
    def __init__(self) -> None:
        self.statements: list = []

    async def execute(self, statement):
        self.statements.append(statement)

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


async def test_assistant_message_keeps_its_provenance_columns(monkeypatch):
    session = RecordingSession()

    @asynccontextmanager
    async def fake_session():
        yield session

    async def owner(self, _session, _conversation_id):
        return 7

    monkeypatch.setattr(conversations, "get_async_session", fake_session)
    monkeypatch.setattr(PgConversationStore, "_owner", owner)

    await PgConversationStore().append(
        7,
        "5e2f7a54-9d4c-4f8e-8f55-0a6d6a4c7b11",
        [
            {"role": "user", "content": "q", "trace_id": "t1"},
            {
                "role": "assistant",
                "content": "a [1]",
                "intent": "learn",
                "citations": [{"n": 1, "paper_id": "2609.20971v1"}],
                "provider": "gemini",
                "trace_id": "t1",
                "agent": "tutor",
                "prompt_version": "tutor_explain@v2",
                "checks": {"grounded": True},
                "trace": {"timings": {"route": 1.2}},
            },
        ],
        title="q",
        state={"mode": "explain"},
    )

    inserts = [s for s in session.statements if isinstance(s, dml.Insert)]
    message_rows = [
        s.compile(dialect=postgresql.dialect()).params
        for s in inserts
        if s.table.name == "messages"
    ]
    assistant = next(p for p in message_rows if p.get("role") == "assistant")
    for column in ("intent", "citations", "provider", "agent", "prompt_version", "checks", "trace"):
        assert assistant.get(column) is not None, f"{column} was not written"
