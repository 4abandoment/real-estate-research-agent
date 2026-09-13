import psycopg
import pytest

from research_agent.config import load_settings
from research_agent.db import (
    ConversationStore,
    apply_schema,
    bump_approval_turns,
    connect,
    create_approval,
    find_approval,
    resolve_approval,
)


@pytest.fixture()
def db():
    database_url = load_settings().database_url
    if not database_url:
        pytest.skip("DATABASE_URL not set")
    try:
        conn = connect(database_url)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    apply_schema(conn)
    yield conn
    conn.execute("DELETE FROM messages WHERE channel_id = 'C_TEST'")
    conn.execute("DELETE FROM pending_approvals WHERE origin_channel_id = 'C_TEST'")


@pytest.fixture()
def store(db):
    return ConversationStore(db)


def test_conversation_roundtrip(store) -> None:
    store.add(channel_id="C_TEST", thread_ts="t1", role="user", content="hi", user_id="U1")
    store.add(channel_id="C_TEST", thread_ts="t1", role="assistant", content="hello")

    history = store.history(channel_id="C_TEST", thread_ts="t1")

    assert [(m.role, m.content) for m in history] == [("user", "hi"), ("assistant", "hello")]


def test_history_is_thread_scoped(store) -> None:
    store.add(channel_id="C_TEST", thread_ts="t1", role="user", content="thread one")
    store.add(channel_id="C_TEST", thread_ts="t2", role="user", content="thread two")

    history = store.history(channel_id="C_TEST", thread_ts="t2")

    assert [m.content for m in history] == ["thread two"]


def test_approval_matches_admin_or_origin_thread(db) -> None:
    create_approval(
        db,
        origin_channel_id="C_TEST",
        origin_thread_ts="t1",
        question="q",
        reason="r",
        admin_thread_ts="a1",
    )

    via_admin = find_approval(db, channel_id="C_ADMIN", thread_ts="a1", admin_channel_id="C_ADMIN")
    via_origin = find_approval(db, channel_id="C_TEST", thread_ts="t1", admin_channel_id="C_ADMIN")
    unmatched = find_approval(db, channel_id="C_ADMIN", thread_ts="zz", admin_channel_id="C_ADMIN")

    assert via_admin is not None and via_origin is not None
    assert via_admin["id"] == via_origin["id"]
    assert via_admin["turns"] == 0
    assert unmatched is None


def test_approval_turns_then_resolve(db) -> None:
    create_approval(
        db,
        origin_channel_id="C_TEST",
        origin_thread_ts="t2",
        question="q",
        reason="r",
        admin_thread_ts="a2",
    )
    approval = find_approval(db, channel_id="C_ADMIN", thread_ts="a2", admin_channel_id="C_ADMIN")

    assert bump_approval_turns(db, approval_id=approval["id"]) == 1

    resolve_approval(db, approval_id=approval["id"], guidance="done")

    assert (
        find_approval(db, channel_id="C_ADMIN", thread_ts="a2", admin_channel_id="C_ADMIN") is None
    )
