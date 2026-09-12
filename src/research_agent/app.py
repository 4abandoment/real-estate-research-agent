"""Slack Bolt app (Socket Mode).

Wires the research agent to Slack: scoping, routing, text-to-SQL reveal,
PII redaction and human-in-the-loop escalation to an admin channel.
"""

import logging

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from research_agent.agent.orchestrator import ResearchAgent
from research_agent.config import Settings, load_settings
from research_agent.db import (
    ConversationStore,
    apply_schema,
    connect,
    create_approval,
    find_approval,
    last_query,
    resolve_approval,
)
from research_agent.embeddings import Embedder
from research_agent.llm.client import AnthropicClient

logger = logging.getLogger(__name__)

FALLBACK = "Research agent online. Ask me about the portfolio."


def build_app(
    settings: Settings,
    *,
    store: ConversationStore | None = None,
    agent: ResearchAgent | None = None,
    conn=None,
) -> App:
    app = App(token=settings.slack_bot_token)

    @app.event("app_mention")
    def on_mention(event, say, client) -> None:
        question = _clean_question(event.get("text", ""))
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")

        if store is not None and channel_id and thread_ts:
            store.add(
                channel_id=channel_id,
                thread_ts=thread_ts,
                role="user",
                content=question,
                user_id=event.get("user"),
            )

        if agent is None or conn is None or not channel_id or not thread_ts:
            say(FALLBACK)
            return

        result = agent.handle(question=question, channel_id=channel_id, thread_ts=thread_ts)

        if result.needs_human and settings.slack_admin_channel_id:
            _escalate(settings, conn, client, channel_id, thread_ts, question, result)
            reply = (
                "This one needs a human reviewer. I've asked in the admin channel "
                "and will follow up here."
            )
        else:
            reply = result.answer

        say(reply)
        if store is not None:
            store.add(channel_id=channel_id, thread_ts=thread_ts, role="assistant", content=reply)

    @app.command("/sql")
    def show_sql(ack, command, respond) -> None:
        ack()
        if conn is None:
            respond("SQL log unavailable.")
            return
        record = last_query(conn, channel_id=command.get("channel_id"))
        if not record or not record["sql"]:
            respond("No SQL has been run in this channel yet.")
            return
        respond(f"Last question: {record['question']}\n```{record['sql']}```")

    @app.event("message")
    def on_admin_reply(event, client) -> None:
        if conn is None or not settings.slack_admin_channel_id:
            return
        if event.get("channel") != settings.slack_admin_channel_id:
            return
        if event.get("bot_id") or event.get("subtype") or not event.get("thread_ts"):
            return
        approval = find_approval(conn, admin_thread_ts=event["thread_ts"])
        if approval is None:
            return
        guidance = event.get("text", "").strip()
        client.chat_postMessage(
            channel=approval["origin_channel_id"],
            thread_ts=approval["origin_thread_ts"],
            text=f"*Reviewer guidance:* {guidance}",
        )
        resolve_approval(conn, admin_thread_ts=event["thread_ts"], guidance=guidance)

    return app


def _escalate(settings, conn, client, channel_id, thread_ts, question, result) -> None:
    reason = result.escalation_reason or "low confidence"
    response = client.chat_postMessage(
        channel=settings.slack_admin_channel_id,
        text=(
            ":bust_in_silhouette: Review needed\n"
            f"*Question:* {question}\n"
            f"*Reason:* {reason}\n"
            "Reply in this thread with guidance."
        ),
    )
    create_approval(
        conn,
        origin_channel_id=channel_id,
        origin_thread_ts=thread_ts,
        question=question,
        reason=reason,
        admin_thread_ts=response["ts"],
    )


def _clean_question(text: str) -> str:
    parts = [part for part in text.split() if not part.startswith("<@")]
    return " ".join(parts).strip() or text.strip()


def _build_store(settings: Settings):
    if not settings.database_url:
        logger.warning("DATABASE_URL not set; conversation history disabled")
        return None, None
    conn = connect(settings.database_url)
    apply_schema(conn)
    return ConversationStore(conn), conn


def main() -> None:
    settings = load_settings()
    logging.basicConfig(level=settings.log_level)

    if not settings.socket_mode:
        raise SystemExit("HTTP transport not implemented yet. Set SOCKET_MODE=true.")
    if not settings.slack_bot_token or not settings.slack_app_token:
        raise SystemExit("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required.")

    store, conn = _build_store(settings)
    agent = None
    if conn is not None and settings.anthropic_api_key:
        agent = ResearchAgent(
            llm=AnthropicClient(settings.anthropic_api_key),
            conn=conn,
            embedder=Embedder(),
            conversation=store,
        )
    else:
        logger.warning("Agent disabled (missing DATABASE_URL or ANTHROPIC_API_KEY)")

    app = build_app(settings, store=store, agent=agent, conn=conn)
    logger.info("Starting Socket Mode handler")
    SocketModeHandler(app, settings.slack_app_token).start()


if __name__ == "__main__":
    main()
