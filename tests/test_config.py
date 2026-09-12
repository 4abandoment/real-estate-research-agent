from research_agent.config import load_settings


def test_reads_tokens_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SOCKET_MODE", "false")

    settings = load_settings()

    assert settings.slack_bot_token == "xoxb-test"
    assert settings.slack_app_token == "xapp-test"
    assert settings.socket_mode is False


def test_socket_mode_defaults_to_true(monkeypatch) -> None:
    monkeypatch.delenv("SOCKET_MODE", raising=False)

    settings = load_settings()

    assert settings.socket_mode is True


def test_blank_values_become_none(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")

    settings = load_settings()

    assert settings.anthropic_api_key is None
