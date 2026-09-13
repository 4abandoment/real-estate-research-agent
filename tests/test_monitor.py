import json
import time

import psycopg
import pytest

from research_agent import monitor


def test_infer_stage_sequences() -> None:
    assert monitor.infer_stage([]) == "queued"
    assert monitor.infer_stage(["scope"]) == "scoping"
    assert monitor.infer_stage(["scope", "sql"]) == "querying warehouse"
    assert monitor.infer_stage(["scope", "sql", "synthesize"]) == "writing answer"


def test_load_usage_entries_skips_bad_lines(tmp_path) -> None:
    path = tmp_path / "usage.jsonl"
    path.write_text('{"ts": 1, "task": "scope"}\nnot json\n{"ts": 3, "task": "sql"}\n')

    entries = monitor.load_usage_entries(path)

    assert [entry["task"] for entry in entries] == ["scope", "sql"]


def test_load_usage_entries_sorts_and_limits(tmp_path) -> None:
    path = tmp_path / "usage.jsonl"
    path.write_text("\n".join(json.dumps({"ts": i, "task": "t"}) for i in (5, 1, 3)) + "\n")

    assert [entry["ts"] for entry in monitor.load_usage_entries(path)] == [1, 3, 5]
    assert [entry["ts"] for entry in monitor.load_usage_entries(path, limit=2)] == [3, 5]


def test_load_usage_entries_missing_file(tmp_path) -> None:
    assert monitor.load_usage_entries(tmp_path / "nope.jsonl") == []


def test_usage_payload_aggregates_and_recent(tmp_path) -> None:
    path = tmp_path / "usage.jsonl"
    lines = [
        json.dumps(
            {
                "ts": i,
                "task": "sql",
                "model": "m1",
                "input_tokens": 10,
                "output_tokens": 5,
                "latency_ms": 100.0,
                "cost_usd": 0.01,
            }
        )
        for i in range(3)
    ] + [
        json.dumps(
            {
                "ts": 9,
                "task": "scope",
                "model": "m2",
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 10.0,
                "cost_usd": 0.0,
            }
        )
    ]
    path.write_text("\n".join(lines) + "\n")

    payload = monitor.usage_payload(monitor.load_usage_entries(path))

    by_model = {model["model"]: model for model in payload["models"]}
    assert by_model["m1"]["calls"] == 3
    assert by_model["m1"]["input_tokens"] == 30
    assert by_model["m2"]["calls"] == 1
    assert payload["recent"][0]["task"] == "scope"


def test_activity_payload_without_db() -> None:
    payload = monitor.activity_payload(None, [], now=1000.0)

    assert payload == {"queries": [], "messages": [], "approvals": [], "in_flight": []}


def test_activity_payload_with_db() -> None:
    from research_agent.config import load_settings
    from research_agent.db import connect

    database_url = load_settings().database_url
    if not database_url:
        pytest.skip("DATABASE_URL not set")
    try:
        conn = connect(database_url)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")

    payload = monitor.activity_payload(conn, [], now=time.time())

    assert isinstance(payload["queries"], list)
    assert isinstance(payload["in_flight"], list)
    assert all(item["label"] in {"user", "eval"} for item in payload["queries"])


def test_status_payload_without_db(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(monitor, "get_conn", lambda: None)
    monkeypatch.setattr(monitor, "bot_running", lambda: (True, 1000))
    monkeypatch.setenv("USAGE_LOG_PATH", str(tmp_path / "u.jsonl"))
    from research_agent.config import load_settings

    payload = monitor.status_payload(load_settings())

    assert payload["bot_running"] is True
    assert payload["bot_uptime_s"] > 0
    assert payload["tables"] == {}
    assert payload["pending_reviews"] is None
    assert payload["calls"] == 0


def test_bot_running_detects_own_invocation() -> None:
    running, created = monitor.bot_running()

    assert isinstance(running, bool)
    if running:
        assert isinstance(created, int)


def test_page_and_api_serve(monkeypatch) -> None:
    monkeypatch.setattr(monitor, "status_payload", lambda settings: {"bot_running": True})
    monkeypatch.setattr(monitor, "activity_payload", lambda conn, entries: {"queries": []})
    monkeypatch.setattr(monitor, "usage_payload", lambda entries: {"models": []})

    client = monitor.app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"Research agent monitor" in page.data

    assert client.get("/api/status").get_json() == {"bot_running": True}
    assert client.get("/api/activity").get_json() == {"queries": []}
    assert client.get("/api/usage").get_json() == {"models": []}
