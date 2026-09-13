from types import SimpleNamespace

import pytest

from research_agent.agent.orchestrator import ResearchAgent
from research_agent.agent.sql import generate_sql
from research_agent.llm.client import LLMResponse


class _ScriptedClient:
    """Returns queued responses in order and records every call."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, *, messages, model, system=None, max_tokens=1024, task="llm"):
        self.calls.append(
            {"task": task, "max_tokens": max_tokens, "messages": messages, "system": system}
        )
        return self.responses.pop(0)


class _FakeCursor:
    description = []

    def fetchall(self):
        return []

    def fetchmany(self, size):
        return []


class _FakeConn:
    def execute(self, *args, **kwargs):
        return _FakeCursor()


class _FakeStore:
    def history(self, *, channel_id, thread_ts, limit=20):
        return []


class _FakeEmbedder:
    @staticmethod
    def to_pgvector(vector):
        return "[0.0]"

    def embed(self, texts):
        return [[0.0] for _ in texts]


def _agent(client) -> ResearchAgent:
    return ResearchAgent(
        llm=client,
        conn=_FakeConn(),
        embedder=_FakeEmbedder(),
        conversation=_FakeStore(),
    )


SYNTH_OK = "ANSWER: All good.\nCONFIDENCE: 0.9\nNEEDS_HUMAN: false\nREASON:"


def test_directive_skips_scope_and_reaches_synthesis() -> None:
    client = _ScriptedClient([LLMResponse(SYNTH_OK, "m", 1, 2, 1.0)])
    agent = _agent(client)

    result = agent.handle(
        question="Should we evict this tenant?",
        channel_id="C",
        thread_ts="T",
        directive="Please decline this request",
    )

    assert [call["task"] for call in client.calls] == ["synthesize"]
    synth_content = client.calls[0]["messages"][0]["content"]
    assert "Reviewer direction (authoritative): Please decline this request" in synth_content
    assert result.answer == "All good."
    assert result.needs_human is False


def test_without_directive_scope_runs() -> None:
    client = _ScriptedClient(
        [
            LLMResponse('{"needs_clarification": false, "questions": []}', "m", 1, 2, 1.0),
            LLMResponse(SYNTH_OK, "m", 1, 2, 1.0),
        ]
    )
    agent = _agent(client)

    agent.handle(question="What is our overdue rent?", channel_id="C", thread_ts="T")

    assert [call["task"] for call in client.calls] == ["scope", "synthesize"]


def test_scope_max_tokens_flow_to_llm() -> None:
    client = _ScriptedClient(
        [
            LLMResponse('{"needs_clarification": false, "questions": []}', "m", 1, 2, 1.0),
            LLMResponse(SYNTH_OK, "m", 1, 2, 1.0),
        ]
    )
    agent = ResearchAgent(
        llm=client,
        conn=_FakeConn(),
        embedder=_FakeEmbedder(),
        conversation=_FakeStore(),
        scope_max_tokens=123,
        synth_max_tokens=456,
    )

    agent.handle(question="q", channel_id="C", thread_ts="T")

    assert client.calls[0]["max_tokens"] == 123
    assert client.calls[1]["max_tokens"] == 456


def test_truncated_synthesis_appends_marker() -> None:
    client = _ScriptedClient(
        [
            LLMResponse('{"needs_clarification": false, "questions": []}', "m", 1, 2, 1.0),
            LLMResponse(SYNTH_OK, "m", 1, 2, 1.0, stop_reason="max_tokens"),
        ]
    )
    agent = _agent(client)

    result = agent.handle(question="q", channel_id="C", thread_ts="T")

    assert result.answer.endswith("\n_(answer may be cut off)_")


def test_truncated_sql_retries_with_doubled_cap() -> None:
    client = _ScriptedClient(
        [
            LLMResponse(
                "SELECT 1 AS a FROM listings LIMIT", "m", 1, 2, 1.0, stop_reason="max_tokens"
            ),
            LLMResponse("SELECT 1 AS a FROM listings LIMIT 5", "m", 1, 2, 1.0),
        ]
    )

    sql = generate_sql(client, "q", max_tokens=1500)

    assert sql == "SELECT 1 AS a FROM listings LIMIT 5"
    assert [call["max_tokens"] for call in client.calls] == [1500, 3000]


def test_untruncated_sql_does_not_retry() -> None:
    client = _ScriptedClient([LLMResponse("SELECT 1 AS a", "m", 1, 2, 1.0)])

    generate_sql(client, "q", max_tokens=1500)

    assert len(client.calls) == 1


def test_scripted_client_exhausted_raises() -> None:
    client = _ScriptedClient([])

    with pytest.raises(IndexError):
        generate_sql(client, "q")


def test_routing_includes_earlier_user_turns(monkeypatch) -> None:
    import research_agent.agent.orchestrator as orchestrator

    captured: list[str] = []
    monkeypatch.setattr(
        orchestrator,
        "route_question",
        lambda conn, embedder, question, top_k=3: captured.append(question) or [],
    )

    class _StoreWithHistory:
        def history(self, *, channel_id, thread_ts, limit=20):
            return [
                SimpleNamespace(
                    role="user",
                    content="What kind of negative reviews are common in high-arrears listings?",
                ),
                SimpleNamespace(role="assistant", content="Before I dig in: ..."),
                SimpleNamespace(
                    role="user", content="listing price, arrears measured by total overdue balance"
                ),
            ]

    client = _ScriptedClient(
        [
            LLMResponse('{"needs_clarification": false, "questions": []}', "m", 1, 2, 1.0),
            LLMResponse(SYNTH_OK, "m", 1, 2, 1.0),
        ]
    )
    agent = ResearchAgent(
        llm=client,
        conn=_FakeConn(),
        embedder=_FakeEmbedder(),
        conversation=_StoreWithHistory(),
    )

    agent.handle(
        question="listing price, arrears measured by total overdue balance",
        channel_id="C",
        thread_ts="T",
    )

    assert "negative reviews" in captured[0]
    assert "total overdue balance" in captured[0]
    assert captured[0].count("total overdue balance") == 1
