from research_agent.llm.client import OPENROUTER_PREFIX, RoutedClient
from research_agent.llm.model_router import CAPABLE, CHEAP, model_for


class _FakeOpenRouter:
    def __init__(self) -> None:
        self.model = None

    def complete(self, *, messages, model, system=None, max_tokens=1024, task="llm"):
        self.model = model
        return None


class _FakeAnthropic:
    def __init__(self) -> None:
        self.model = None

    def complete(self, *, messages, model, system=None, max_tokens=1024, task="llm"):
        self.model = model
        return None


def test_model_for_defaults(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_SCOPE", raising=False)
    monkeypatch.delenv("MODEL_SQL", raising=False)
    monkeypatch.delenv("MODEL_SYNTH", raising=False)

    assert model_for("scope") == CHEAP
    assert model_for("sql") == CAPABLE
    assert model_for("synthesize") == CAPABLE


def test_model_for_env_override(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_SQL", raising=False)
    monkeypatch.setenv("MODEL_SYNTH", "openrouter/deepseek/deepseek-v4.1-flash")

    assert model_for("synthesize") == "openrouter/deepseek/deepseek-v4.1-flash"
    assert model_for("sql") == CAPABLE


def test_routed_client_dispatches_by_prefix() -> None:
    openrouter = _FakeOpenRouter()
    anthropic = _FakeAnthropic()
    client = RoutedClient(anthropic, openrouter)

    client.complete(
        messages=[{"role": "user", "content": "q"}],
        model=f"{OPENROUTER_PREFIX}deepseek/deepseek-v4.1-flash",
    )
    client.complete(messages=[{"role": "user", "content": "q"}], model="claude-sonnet-5")

    assert openrouter.model == "openrouter/deepseek/deepseek-v4.1-flash"
    assert anthropic.model == "claude-sonnet-5"


def test_routed_client_requires_key_for_openrouter() -> None:
    import pytest

    client = RoutedClient(_FakeAnthropic(), None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model=f"{OPENROUTER_PREFIX}deepseek/deepseek-v4.1-flash",
        )


def test_routed_client_dispatches_opencode_by_prefix() -> None:
    opencode = _FakeOpenRouter()
    client = RoutedClient(_FakeAnthropic(), None, opencode)

    client.complete(
        messages=[{"role": "user", "content": "q"}],
        model="opencode/deepseek-v4-flash",
    )

    assert opencode.model == "opencode/deepseek-v4-flash"


def test_routed_client_requires_key_for_opencode() -> None:
    import pytest

    client = RoutedClient(_FakeAnthropic(), None)
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        client.complete(
            messages=[{"role": "user", "content": "q"}],
            model="opencode/deepseek-v4-flash",
        )
