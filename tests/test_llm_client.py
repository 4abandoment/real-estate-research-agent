import io
import json
import urllib.error

import pytest

from research_agent.llm import client as client_module


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False


@pytest.fixture(autouse=True)
def _silence_usage_log(monkeypatch):
    monkeypatch.setattr(client_module, "log_usage", lambda *args, **kwargs: None)


def _patch_urlopen(monkeypatch, payloads: list[dict]) -> dict:
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        payload = payloads[min(calls["count"], len(payloads) - 1)]
        calls["count"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_openrouter_error_body_raises_with_message(monkeypatch) -> None:
    _patch_urlopen(monkeypatch, [{"error": {"message": "insufficient credits", "code": 402}}])

    with pytest.raises(RuntimeError, match="insufficient credits"):
        client_module.OpenRouterClient("sk-test").complete(
            messages=[{"role": "user", "content": "q"}],
            model="openrouter/deepseek/deepseek-v4.1-flash",
        )


def test_openrouter_missing_choices_raises(monkeypatch) -> None:
    _patch_urlopen(monkeypatch, [{"id": "gen-1"}])

    with pytest.raises(RuntimeError, match="no choices"):
        client_module.OpenRouterClient("sk-test").complete(
            messages=[{"role": "user", "content": "q"}],
            model="openrouter/deepseek/deepseek-v4.1-flash",
        )


def test_openrouter_http_error_includes_body(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"invalid api key"}}'),
        )

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="invalid api key"):
        client_module.OpenRouterClient("sk-test").complete(
            messages=[{"role": "user", "content": "q"}],
            model="openrouter/deepseek/deepseek-v4.1-flash",
        )


def test_openrouter_retries_transient_error_body(monkeypatch) -> None:
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)
    calls = _patch_urlopen(
        monkeypatch,
        [
            {"error": {"message": "upstream down", "code": 502}},
            {
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ],
    )

    result = client_module.OpenRouterClient("sk-test").complete(
        messages=[{"role": "user", "content": "q"}],
        model="openrouter/deepseek/deepseek-v4.1-flash",
    )

    assert result.text == "hello"
    assert calls["count"] == 2
