from dataclasses import replace
from types import SimpleNamespace

import psycopg

from research_agent import app as app_module
from research_agent.agent.orchestrator import AgentResult
from research_agent.app import (
    SLACK_TEXT_LIMIT,
    _chunk_text,
    _deliver,
    _failure_notice,
    _respond,
    _sql_with_rows,
)
from research_agent.config import load_settings


class _FakeClient:
    def __init__(self, fail_on=None) -> None:
        self.calls = []
        self._fail_on = fail_on

    def chat_postMessage(self, **kwargs):
        self.calls.append(("post", kwargs))
        return {"ts": "1.1"}

    def chat_update(self, **kwargs):
        text = kwargs.get("text", "")
        self.calls.append(("update", kwargs))
        if self._fail_on and self._fail_on(text):
            raise RuntimeError("msg_too_long")
        return {"ts": kwargs.get("ts")}

    def reactions_add(self, **kwargs):
        self.calls.append(("react", kwargs))

    def reactions_remove(self, **kwargs):
        self.calls.append(("unreact", kwargs))


class _FakeAgent:
    def __init__(self, result: AgentResult) -> None:
        self._result = result

    def handle(self, *, question, channel_id, thread_ts, progress=None, directive=None):
        return self._result


class _RaisingAgent:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def handle(self, *, question, channel_id, thread_ts, progress=None, directive=None):
        raise self._error


class _FakeStore:
    def __init__(self) -> None:
        self.added = []

    def add(self, **kwargs):
        self.added.append(kwargs)


def test_chunk_short_text_single_chunk() -> None:
    assert _chunk_text("hello world") == ["hello world"]


def test_chunk_exact_limit_single_chunk() -> None:
    text = "x" * SLACK_TEXT_LIMIT
    assert _chunk_text(text) == [text]


def test_chunk_long_lines_never_exceed_limit() -> None:
    text = "\n".join(f"line {i}" for i in range(2000))
    chunks = _chunk_text(text, limit=100)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "\n".join(chunks) == text


def test_chunk_hard_splits_single_overlong_line() -> None:
    text = "A" * 5000
    chunks = _chunk_text(text, limit=100)
    assert all(len(chunk) == 100 for chunk in chunks)
    assert "".join(chunks) == text


def test_chunk_preserves_paragraph_breaks() -> None:
    text = "para one\n\npara two\n\npara three"
    chunks = _chunk_text(text, limit=20)
    assert "\n".join(chunks) == "\n".join(text.splitlines())
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_deliver_short_reply_updates_placeholder_only() -> None:
    client = _FakeClient()
    _deliver(client, "C", "T", "1.1", "short reply")
    updates = [c for c in client.calls if c[0] == "update"]
    posts = [c for c in client.calls if c[0] == "post"]
    assert len(updates) == 1 and updates[0][1]["text"] == "short reply"
    assert posts == []


def test_deliver_long_reply_chunks_across_messages() -> None:
    client = _FakeClient()
    reply = "line " * 3000
    _deliver(client, "C", "T", "1.1", reply)
    updates = [c for c in client.calls if c[0] == "update"]
    posts = [c for c in client.calls if c[0] == "post"]
    chunks = _chunk_text(reply)
    assert len(updates) == 1
    assert updates[0][1]["text"] == chunks[0]
    assert [p[1]["text"] for p in posts] == chunks[1:]
    assert all(p[1]["thread_ts"] == "T" for p in posts)


def _answer_posts(client) -> list:
    return [
        c[1]["text"]
        for c in client.calls
        if c[0] == "post" and not c[1]["text"].startswith("Researching")
    ]


def test_respond_long_answer_posts_and_stores_full_reply() -> None:
    client = _FakeClient()
    store = _FakeStore()
    answer = "paragraph about arrears and guests.\n\n" * 300
    _respond(
        load_settings(),
        conn=object(),
        agent=_FakeAgent(AgentResult(answer=answer, sources=["transactions_ledger"])),
        store=store,
        client=client,
        question="What are the top arrears?",
        channel_id="C",
        thread_ts="T",
        message_ts=None,
    )
    updates = [c for c in client.calls if c[0] == "update"]
    chunks = _chunk_text(answer + "\n_Sources: transactions_ledger_")
    assert updates[0][1]["text"] == chunks[0]
    assert _answer_posts(client) == chunks[1:]
    assert store.added[-1]["content"] == answer + "\n_Sources: transactions_ledger_"


def test_respond_short_answer_single_update() -> None:
    client = _FakeClient()
    store = _FakeStore()
    _respond(
        load_settings(),
        conn=object(),
        agent=_FakeAgent(AgentResult(answer="All quiet.")),
        store=store,
        client=client,
        question="Anything interesting?",
        channel_id="C",
        thread_ts="T",
        message_ts=None,
    )
    updates = [c for c in client.calls if c[0] == "update"]
    assert updates[-1][1]["text"] == "All quiet."
    assert _answer_posts(client) == []
    assert store.added[-1]["content"] == "All quiet."


def test_respond_delivery_failure_posts_notice_no_exception() -> None:
    client = _FakeClient(fail_on=lambda text: text.startswith("A"))
    store = _FakeStore()
    answer = "A" * 5000
    _respond(
        load_settings(),
        conn=object(),
        agent=_FakeAgent(AgentResult(answer=answer)),
        store=store,
        client=client,
        question="Fail me",
        channel_id="C",
        thread_ts="T",
        message_ts=None,
    )
    texts = [c[1]["text"] for c in client.calls if c[0] == "update"]
    assert any("failed to post" in text for text in texts)
    assert all(item["role"] != "assistant" for item in store.added)


def test_failure_notice_classifies_causes() -> None:
    text, infra = _failure_notice(psycopg.OperationalError("db down"), "R1")
    assert "warehouse" in text and infra is True and "R1" in text

    text, infra = _failure_notice(OSError("timed out"), "R1")
    assert "model" in text.lower() and infra is True

    text, infra = _failure_notice(RuntimeError("boom"), "R1")
    assert "Something went wrong" in text and infra is False


def test_respond_infra_failure_alerts_admin_once(monkeypatch) -> None:
    app_module._ALERTED.clear()
    # Hermetic: never depend on a local .env for the admin channel (CI has none).
    monkeypatch.setattr(
        app_module,
        "load_settings",
        lambda: replace(load_settings(), slack_admin_channel_id="C_ADMIN"),
    )
    settings = load_settings()
    error = psycopg.OperationalError("db down")
    client = _FakeClient()
    _respond(
        settings,
        conn=object(),
        agent=_RaisingAgent(error),
        store=_FakeStore(),
        client=client,
        question="q",
        channel_id="C",
        thread_ts="T",
        message_ts=None,
    )
    texts = [c[1]["text"] for c in client.calls if c[0] == "update"]
    assert any("warehouse" in text for text in texts)

    def _admin_posts(fake: _FakeClient) -> list:
        return [
            c
            for c in fake.calls
            if c[0] == "post" and c[1].get("channel") == settings.slack_admin_channel_id
        ]

    assert len(_admin_posts(client)) == 1

    second = _FakeClient()
    _respond(
        settings,
        conn=object(),
        agent=_RaisingAgent(error),
        store=_FakeStore(),
        client=second,
        question="q",
        channel_id="C",
        thread_ts="T",
        message_ts=None,
    )
    assert _admin_posts(second) == []


class _RowsCursor:
    description = [SimpleNamespace(name="listing_id"), SimpleNamespace(name="overdue_gbp")]

    def fetchmany(self, size):
        return [(13254774, 300045)]


class _RowsConn:
    def execute(self, *args, **kwargs):
        return _RowsCursor()


def test_sql_with_rows_includes_result_rows() -> None:
    record = {
        "question": "What is our overdue rent?",
        "sql": "SELECT listing_id, overdue_gbp FROM transactions",
    }

    text = _sql_with_rows(_RowsConn(), record)

    assert "Result (top rows)" in text
    assert "13254774" in text and "300045" in text


def test_sql_with_rows_falls_back_on_run_error() -> None:
    class _BoomConn:
        def execute(self, *args, **kwargs):
            raise RuntimeError("boom")

    record = {"question": "q", "sql": "SELECT listing_id FROM transactions"}

    text = _sql_with_rows(_BoomConn(), record)

    assert "```" in text and "Result" not in text
