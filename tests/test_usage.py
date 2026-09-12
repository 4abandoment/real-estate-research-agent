import json

from research_agent.llm.client import LLMResponse
from research_agent.llm.usage import estimate_cost, log_usage, track_usage


def _response() -> LLMResponse:
    return LLMResponse(
        text="ok",
        model="claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=12.0,
    )


def test_estimate_cost() -> None:
    assert estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == 6.0
    assert estimate_cost("unknown-model", 999, 999) == 0.0


def test_log_usage_writes_jsonl(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("USAGE_LOG_PATH", str(log_file))

    entry = log_usage("sql", _response())

    assert entry.cost_usd == 6.0
    written = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert written["task"] == "sql"
    assert written["model"] == "claude-haiku-4-5-20251001"


def test_track_usage_decorator_logs(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("USAGE_LOG_PATH", str(log_file))

    @track_usage("synthesize")
    def call() -> LLMResponse:
        return _response()

    call()

    assert log_file.exists()
    assert json.loads(log_file.read_text(encoding="utf-8").strip())["task"] == "synthesize"
