from research_agent.config import load_settings


def test_reads_tokens_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SOCKET_MODE", "false")

    settings = load_settings()

    assert settings.slack_bot_token == "xoxb-test"
    assert settings.slack_app_token == "xapp-test"
    assert settings.socket_mode is False


def test_reads_opencode_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-zen-test")

    assert load_settings().opencode_api_key == "sk-zen-test"


def test_socket_mode_defaults_to_true(monkeypatch) -> None:
    monkeypatch.delenv("SOCKET_MODE", raising=False)

    settings = load_settings()

    assert settings.socket_mode is True


def test_blank_values_become_none(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")

    settings = load_settings()

    assert settings.anthropic_api_key is None


def test_token_caps_have_defaults(monkeypatch) -> None:
    for name in ("SCOPE_MAX_TOKENS", "SQL_MAX_TOKENS", "SYNTH_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.scope_max_tokens == 500
    assert settings.sql_max_tokens == 3000
    assert settings.synth_max_tokens == 3000


def test_token_caps_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SCOPE_MAX_TOKENS", "400")
    monkeypatch.setenv("SQL_MAX_TOKENS", "4096")
    monkeypatch.setenv("SYNTH_MAX_TOKENS", "2048")

    settings = load_settings()

    assert settings.scope_max_tokens == 400
    assert settings.sql_max_tokens == 4096
    assert settings.synth_max_tokens == 2048


def test_invalid_token_cap_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("SQL_MAX_TOKENS", "not-a-number")

    settings = load_settings()

    assert settings.sql_max_tokens == 3000
