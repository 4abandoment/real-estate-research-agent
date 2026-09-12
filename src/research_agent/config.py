"""Application settings loaded from the environment.

Secrets live in `.env` (gitignored). Real OS environment variables win, so the
same code works locally and in the cloud.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=False)


def _text(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _flag(name: str, default: bool) -> bool:
    value = _text(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    slack_bot_token: str | None
    slack_app_token: str | None
    slack_admin_channel_id: str | None
    anthropic_api_key: str | None
    database_url: str | None
    socket_mode: bool
    log_level: str
    usage_log_path: str


def load_settings() -> Settings:
    return Settings(
        slack_bot_token=_text("SLACK_BOT_TOKEN"),
        slack_app_token=_text("SLACK_APP_TOKEN"),
        slack_admin_channel_id=_text("SLACK_ADMIN_CHANNEL_ID"),
        anthropic_api_key=_text("ANTHROPIC_API_KEY"),
        database_url=_text("DATABASE_URL"),
        socket_mode=_flag("SOCKET_MODE", default=True),
        log_level=_text("LOG_LEVEL") or "INFO",
        usage_log_path=_text("USAGE_LOG_PATH") or "logs/usage.jsonl",
    )
