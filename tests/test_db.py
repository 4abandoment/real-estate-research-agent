import psycopg
import pytest

from research_agent.config import load_settings
from research_agent.db import ConversationStore, apply_schema, connect


@pytest.fixture()
def store():
    database_url = load_settings().database_url
    if not database_url:
        pytest.skip("DATABASE_URL not set")
    try:
        conn = connect(database_url)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    apply_schema(conn)
    yield ConversationStore(conn)
    conn.execute("DELETE FROM messages WHERE channel_id = 'C_TEST'")


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
