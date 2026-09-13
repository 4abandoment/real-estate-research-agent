"""Application settings loaded from the environment.

Secrets live in `.env` (gitignored) and that file is the source of truth
locally; `override=True` stops stale OS environment variables shadowing it
(which silently pinned an old API key once). In the cloud there is no `.env`,
so real environment variables are used unchanged.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


def _text(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _flag(name: str, default: bool) -> bool:
    value = _text(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = _text(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    slack_bot_token: str | None
    slack_app_token: str | None
    slack_admin_channel_id: str | None
    slack_config_token: str | None
    slack_config_refresh_token: str | None
    slack_app_id: str | None
    anthropic_api_key: str | None
    openrouter_api_key: str | None
    opencode_api_key: str | None
    database_url: str | None
    socket_mode: bool
    log_level: str
    usage_log_path: str
    scope_max_tokens: int
    sql_max_tokens: int
    synth_max_tokens: int


def load_settings() -> Settings:
    return Settings(
        slack_bot_token=_text("SLACK_BOT_TOKEN"),
        slack_app_token=_text("SLACK_APP_TOKEN"),
        slack_admin_channel_id=_text("SLACK_ADMIN_CHANNEL_ID"),
        slack_config_token=_text("SLACK_CONFIG_TOKEN"),
        slack_config_refresh_token=_text("SLACK_CONFIG_REFRESH_TOKEN"),
        slack_app_id=_text("SLACK_APP_ID"),
        anthropic_api_key=_text("ANTHROPIC_API_KEY"),
        openrouter_api_key=_text("OPENROUTER_API_KEY"),
        opencode_api_key=_text("OPENCODE_API_KEY"),
        database_url=_text("DATABASE_URL"),
        socket_mode=_flag("SOCKET_MODE", default=True),
        log_level=_text("LOG_LEVEL") or "INFO",
        usage_log_path=_text("USAGE_LOG_PATH") or "logs/usage.jsonl",
        scope_max_tokens=_int("SCOPE_MAX_TOKENS", default=500),
        sql_max_tokens=_int("SQL_MAX_TOKENS", default=3000),
        synth_max_tokens=_int("SYNTH_MAX_TOKENS", default=3000),
    )
