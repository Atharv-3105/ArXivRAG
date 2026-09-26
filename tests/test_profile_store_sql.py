"""PgProfileStore / PgConversationStore SQL semantics, compiled against the Postgres dialect
(the same approach as test_repositories.py: no database needed)."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from src.core.exceptions import NotFoundError
from src.memory import store as store_mod
from src.memory.model import Changes, MasteryState
from src.memory.store import ConcurrentUpdate, PgProfileStore
from src.storage import conversations as conv_mod

NOW = datetime(2026, 9, 26, tzinfo=UTC)


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def _session(results):
    session = AsyncMock()
    session.execute.side_effect = results
    return session


def _use(monkeypatch, module, session):
    @asynccontextmanager
    async def fake():
        yield session

    monkeypatch.setattr(module, "get_async_session", fake)


def _result(scalar=None, rowcount=1, rows=()):
    r = MagicMock()
    r.scalar_one_or_none.return_value = scalar
    r.scalar_one.return_value = scalar
    r.rowcount = rowcount
    r.scalars.return_value = list(rows)
    r.one_or_none.return_value = scalar
    return r


def _profile_row(version=3):
    row = MagicMock()
    row.profile, row.locked, row.version, row.updated_at = {}, [], version, NOW
    return row


async def test_duplicate_event_stops_after_the_idempotency_insert(monkeypatch):
    session = _session([_result(scalar=None)])  # ON CONFLICT DO NOTHING returned no row
    _use(monkeypatch, store_mod, session)
    applied = await PgProfileStore().apply_event(1, "e1", "chat_turn", {}, lambda s: Changes())
    assert applied is False
    assert session.execute.await_count == 1
    assert "ON CONFLICT (event_id) DO NOTHING" in _sql(session.execute.await_args.args[0])
    session.commit.assert_not_awaited()


async def test_event_writes_are_guarded_and_committed_once(monkeypatch):
    changes = Changes(
        mastery=[MasteryState("lora", 0.4, NOW, evidence=1)],
        profile_patch={"inferred_interests": {"lora": 1.0}},
    )
    session = _session(
        [
            _result(scalar="e1"),  # event inserted
            _result(),  # profile row ensure
            _result(scalar=_profile_row(version=3)),  # profile select
            _result(rows=[]),  # mastery rows
            _result(),  # mastery upsert
            _result(rowcount=1),  # profile patch
        ]
    )
    _use(monkeypatch, store_mod, session)
    assert await PgProfileStore().apply_event(1, "e1", "chat_turn", {}, lambda s: changes) is True

    upsert = _sql(session.execute.await_args_list[4].args[0])
    assert "ON CONFLICT (user_id, concept) DO UPDATE" in upsert
    assert "WHERE concept_mastery.locked IS false" in upsert  # never overwrite a user's lock
    patch = _sql(session.execute.await_args_list[5].args[0])
    assert "user_profiles.profile || " in patch  # atomic JSONB merge in the database
    assert "user_profiles.version = " in patch and "version=(user_profiles.version + " in patch
    session.commit.assert_awaited_once()


async def test_version_mismatch_raises_concurrent_update_without_commit(monkeypatch):
    changes = Changes(profile_patch={"expertise_level": "advanced"})
    session = _session(
        [
            _result(scalar="e1"),
            _result(),
            _result(scalar=_profile_row()),
            _result(rows=[]),
            _result(rowcount=0),  # someone bumped the version in between
        ]
    )
    _use(monkeypatch, store_mod, session)
    with pytest.raises(ConcurrentUpdate):
        await PgProfileStore().apply_event(1, "e1", "chat_turn", {}, lambda s: changes)
    session.commit.assert_not_awaited()  # the event row rolls back with everything else


async def test_foreign_conversation_is_not_found_and_nothing_is_written(monkeypatch):
    owner = MagicMock(user_id=2)
    session = _session([_result(), _result(scalar=owner)])
    _use(monkeypatch, conv_mod, session)
    with pytest.raises(NotFoundError):
        await conv_mod.PgConversationStore().append(1, "c1", [{"role": "user", "content": "x"}])
    session.commit.assert_not_awaited()
    assert session.execute.await_count == 2  # no message insert


async def test_daily_quota_counts_only_this_users_questions(monkeypatch):
    session = _session([_result(scalar=7)])
    _use(monkeypatch, conv_mod, session)
    n = await conv_mod.PgConversationStore().count_user_messages_since(1, NOW)
    sql = _sql(session.execute.await_args.args[0])
    assert n == 7
    assert "conversations.user_id = " in sql and "messages.role = " in sql
    assert "messages.created_at >= " in sql
