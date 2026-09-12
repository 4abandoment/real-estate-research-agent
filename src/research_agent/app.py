"""Slack Bolt app (Socket Mode).

Skeleton only: the agent pipeline is wired in later steps. This proves the
transport, tokens and handlers work end to end.
"""

import logging

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from research_agent.config import Settings, load_settings

logger = logging.getLogger(__name__)


def build_app(settings: Settings) -> App:
    app = App(token=settings.slack_bot_token)

    @app.event("app_mention")
    def on_mention(say) -> None:
        say("Research agent online. Ask me about the portfolio.")

    @app.command("/sql")
    def show_sql(ack, respond) -> None:
        ack()
        respond("No queries run yet.")

    return app


def main() -> None:
    settings = load_settings()
    logging.basicConfig(level=settings.log_level)

    if not settings.socket_mode:
        raise SystemExit("HTTP transport not implemented yet. Set SOCKET_MODE=true.")
    if not settings.slack_bot_token or not settings.slack_app_token:
        raise SystemExit("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required.")

    app = build_app(settings)
    logger.info("Starting Socket Mode handler")
    SocketModeHandler(app, settings.slack_app_token).start()


if __name__ == "__main__":
    main()
