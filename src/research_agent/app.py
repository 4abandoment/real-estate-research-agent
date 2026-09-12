"""Slack Bolt app (Socket Mode).

Skeleton: the agent pipeline is wired in later steps. Records conversation
history so the agent can maintain context across mentions and threads.
"""

import logging

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from research_agent.config import Settings, load_settings
from research_agent.db import ConversationStore, apply_schema, connect

logger = logging.getLogger(__name__)


def build_app(settings: Settings, store: ConversationStore | None = None) -> App:
    app = App(token=settings.slack_bot_token)

    @app.event("app_mention")
    def on_mention(event, say) -> None:
        reply = "Research agent online. Ask me about the portfolio."
        if store is not None:
            _record(store, event, reply)
        say(reply)

    @app.command("/sql")
    def show_sql(ack, respond) -> None:
        ack()
        respond("No queries run yet.")

    return app


def _record(store: ConversationStore, event: dict, reply: str) -> None:
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")
    if not channel_id or not thread_ts:
        return
    store.add(
        channel_id=channel_id,
        thread_ts=thread_ts,
        role="user",
        content=event.get("text", ""),
        user_id=event.get("user"),
    )
    store.add(channel_id=channel_id, thread_ts=thread_ts, role="assistant", content=reply)


def _build_store(settings: Settings) -> ConversationStore | None:
    if not settings.database_url:
        logger.warning("DATABASE_URL not set; conversation history disabled")
        return None
    conn = connect(settings.database_url)
    apply_schema(conn)
    return ConversationStore(conn)


def main() -> None:
    settings = load_settings()
    logging.basicConfig(level=settings.log_level)

    if not settings.socket_mode:
        raise SystemExit("HTTP transport not implemented yet. Set SOCKET_MODE=true.")
    if not settings.slack_bot_token or not settings.slack_app_token:
        raise SystemExit("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required.")

    app = build_app(settings, store=_build_store(settings))
    logger.info("Starting Socket Mode handler")
    SocketModeHandler(app, settings.slack_app_token).start()


if __name__ == "__main__":
    main()
