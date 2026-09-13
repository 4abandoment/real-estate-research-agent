"""Slack Bolt app (Socket Mode).

Wires the research agent to Slack: scoping, routing, text-to-SQL reveal,
PII redaction and human-in-the-loop escalation to an admin channel.

Roles: the user owns question intent (clarifications go back to them, in
thread); the admin reviewer owns judgement (their guidance is acted on and
the outcome is summarised back to the user, attributed).

Progress is proactive: a placeholder message is posted immediately, edited in
place at each pipeline stage, and finished with the answer; the user's message
gets a :eyes: reaction while working and :white_check_mark: when done.
"""

import logging
import re
import time
from pathlib import Path

import anthropic
import psycopg
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from research_agent.agent.orchestrator import AgentResult, ResearchAgent
from research_agent.config import Settings, load_settings
from research_agent.db import (
    WAREHOUSE_TABLES,
    ConversationStore,
    apply_schema,
    bump_approval_turns,
    connect,
    create_approval,
    find_approval,
    last_query,
    resolve_approval,
)
from research_agent.embeddings import Embedder
from research_agent.llm.client import create_client
from research_agent.llm.usage import usage_totals

logger = logging.getLogger(__name__)

FALLBACK = "Research agent online. Ask me about the portfolio."
MAX_REVIEW_TURNS = 4
SQL_FOLLOWUP = re.compile(r"\bsql\b", re.IGNORECASE)
# Slack chat.update rejects text over 4,000 chars (msg_too_long); keep a margin.
SLACK_TEXT_LIMIT = 3900

# ponytail: process-wide alert throttle; per-thread if bursts matter.
_ALERTED: dict[str, float] = {}

# ponytail: single-process bot, so in-memory in-flight tracking is enough.
_ACTIVE: dict[str, dict] = {}
_DURATIONS: list[float] = []


def _note_active(channel_id: str, question: str) -> None:
    _ACTIVE[channel_id] = {"question": question, "started": time.time()}


def _clear_active(channel_id: str) -> None:
    run = _ACTIVE.pop(channel_id, None)
    if run:
        _DURATIONS.append(time.time() - run["started"])
        del _DURATIONS[:-20]


def _typical_seconds() -> float | None:
    return sum(_DURATIONS) / len(_DURATIONS) if _DURATIONS else None


def _status_text(conn, settings: Settings, channel_id: str | None) -> str:
    counts = {
        name: max(int(tuples), 0)
        for name, tuples in conn.execute(
            "SELECT relname, reltuples FROM pg_class WHERE relname = ANY(%s)",
            (list(WAREHOUSE_TABLES),),
        ).fetchall()
    }
    parts = [
        "*Research agent status*",
        "Data loaded (approx): "
        + " · ".join(f"{name} {counts[name]:,}" for name in WAREHOUSE_TABLES if counts.get(name)),
    ]
    pending = conn.execute(
        "SELECT count(*) FROM pending_approvals WHERE status = 'pending'"
    ).fetchone()[0]
    parts.append(f"Pending reviews: {pending}")
    typical = _typical_seconds()
    run = _ACTIVE.get(channel_id or "")
    if run:
        elapsed = time.time() - run["started"]
        suffix = f", typical ~{typical:.0f}s" if typical else ""
        parts.append(f"In progress: {run['question'][:60]} — {elapsed:.0f}s elapsed{suffix}")
    elif typical:
        parts.append(f"Idle — typical run ~{typical:.0f}s")
    calls, cost = usage_totals(Path(settings.usage_log_path))
    parts.append(f"Model calls to date: {calls} (est. ${cost:.2f})")
    if channel_id:
        recent = conn.execute(
            "SELECT left(question, 60), to_char(created_at, 'DD Mon HH24:MI')"
            " FROM query_log WHERE channel_id = %s"
            " ORDER BY created_at DESC LIMIT 5",
            (channel_id,),
        ).fetchall()
        if recent:
            parts.append(
                "Recent questions here:\n"
                + "\n".join(f"• {question} ({when})" for question, when in recent)
            )
    return "\n".join(parts)


def build_app(
    settings: Settings,
    *,
    store: ConversationStore | None = None,
    agent: ResearchAgent | None = None,
    conn=None,
) -> App:
    app = App(token=settings.slack_bot_token)

    @app.event("app_mention")
    def on_mention(event, client) -> None:
        question = _clean_question(event.get("text", ""))
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")
        if not channel_id or not thread_ts:
            return
        _respond(
            settings,
            conn=conn,
            agent=agent,
            store=store,
            client=client,
            question=question,
            channel_id=channel_id,
            thread_ts=thread_ts,
            message_ts=event.get("ts"),
            user_id=event.get("user"),
        )

    @app.command("/sql")
    def show_sql(ack, command, respond, client) -> None:
        ack()
        if conn is None:
            respond("SQL log unavailable.")
            return
        record = last_query(conn, channel_id=command.get("channel_id"))
        if not record or not record["sql"]:
            respond("No SQL has been run in this channel yet.")
            return
        text = f"Last question: {record['question']}\n```{record['sql']}```"
        # Reply in the thread the query ran in; slash-command responds land
        # unthreaded in the channel root and are easy to miss.
        if record.get("thread_ts"):
            try:
                client.chat_postMessage(
                    channel=command.get("channel_id"), thread_ts=record["thread_ts"], text=text
                )
                return
            except Exception as error:  # noqa: BLE001 - fall back to an ephemeral reply
                logger.warning("threaded /sql reply failed: %s", error)
        respond(text)

    @app.command("/progress")
    def show_progress(ack, command, respond) -> None:
        ack()
        if conn is None:
            respond("Status unavailable (no database connection).")
            return
        respond(_status_text(conn, settings, command.get("channel_id")))

    @app.event("message")
    def on_message(event, client) -> None:
        if conn is None:
            return
        if event.get("bot_id") or event.get("subtype"):
            return
        text = (event.get("text") or "").strip()
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts")
        if not text or not channel_id:
            return

        if settings.slack_admin_channel_id and channel_id == settings.slack_admin_channel_id:
            _handle_admin_reply(
                settings,
                conn=conn,
                agent=agent,
                store=store,
                client=client,
                text=text,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )
            return

        # Follow-up in a thread the bot participated in (no @mention needed).
        if agent is None or store is None or not thread_ts or "<@" in text:
            return
        if not store.history(channel_id=channel_id, thread_ts=thread_ts, limit=1):
            return
        _respond(
            settings,
            conn=conn,
            agent=agent,
            store=store,
            client=client,
            question=_clean_question(text),
            channel_id=channel_id,
            thread_ts=thread_ts,
            message_ts=event.get("ts"),
            user_id=event.get("user"),
        )

    return app


def _respond(
    settings,
    *,
    conn,
    agent,
    store,
    client,
    question,
    channel_id,
    thread_ts,
    message_ts=None,
    user_id=None,
) -> None:
    if store is not None:
        store.add(
            channel_id=channel_id,
            thread_ts=thread_ts,
            role="user",
            content=question,
            user_id=user_id,
        )

    # "Show me the SQL" follow-ups answer straight from the query log.
    if conn is not None and SQL_FOLLOWUP.search(question):
        record = last_query(conn, channel_id=channel_id, thread_ts=thread_ts)
        if record is not None:
            if record["sql"]:
                reply = f"Last question: {record['question']}\n```{record['sql']}```"
            else:
                reply = "No SQL was generated for the last question in this thread."
            _post(client, channel_id, thread_ts, reply)
            if store is not None:
                store.add(
                    channel_id=channel_id, thread_ts=thread_ts, role="assistant", content=reply
                )
            return

    if agent is None or conn is None:
        _post(client, channel_id, thread_ts, FALLBACK)
        return

    _react(client, channel_id, message_ts, "eyes")
    placeholder = client.chat_postMessage(
        channel=channel_id, thread_ts=thread_ts, text="Researching… :eyes:"
    )
    _note_active(channel_id, question)

    def progress(stage: str) -> None:
        try:
            client.chat_update(channel=channel_id, ts=placeholder["ts"], text=f"{stage}… :eyes:")
        except Exception as error:  # noqa: BLE001 - progress is best-effort
            logger.warning("progress update failed: %s", error)

    try:
        result = agent.handle(
            question=question, channel_id=channel_id, thread_ts=thread_ts, progress=progress
        )
    except Exception as error:  # noqa: BLE001 - surface failures instead of silent threads
        logger.exception("agent run failed: %s", error)
        _clear_active(channel_id)
        _fail(settings, client, channel_id, placeholder["ts"], error, thread_ts)
        _unreact(client, channel_id, message_ts, "eyes")
        return
    _clear_active(channel_id)

    try:
        if result.needs_human and settings.slack_admin_channel_id:
            _escalate(settings, conn, client, channel_id, thread_ts, question, result)
            reply = (
                "This one needs a human reviewer. I've asked in the admin channel "
                "and will follow up here."
            )
        else:
            reply = _format_reply(result)

        _deliver(client, channel_id, thread_ts, placeholder["ts"], reply)
        _unreact(client, channel_id, message_ts, "eyes")
        _react(client, channel_id, message_ts, "white_check_mark")
        if store is not None:
            store.add(channel_id=channel_id, thread_ts=thread_ts, role="assistant", content=reply)
    except Exception as error:  # noqa: BLE001 - never strand a thread silently
        logger.exception("delivery failed for question %r: %s", question, error)
        _delivery_failed(client, channel_id, placeholder["ts"])


def _handle_admin_reply(
    settings,
    *,
    conn,
    agent,
    store,
    client,
    text,
    channel_id,
    thread_ts,
) -> None:
    approval = (
        find_approval(
            conn,
            channel_id=channel_id,
            thread_ts=thread_ts,
            admin_channel_id=settings.slack_admin_channel_id,
        )
        if thread_ts
        else None
    )
    logger.info("admin reply: thread_ts=%s matched=%s", thread_ts, approval is not None)
    if approval is None:
        return

    origin_channel = approval["origin_channel_id"]
    origin_thread = approval["origin_thread_ts"]

    if approval["turns"] >= MAX_REVIEW_TURNS or agent is None or store is None:
        _post(
            client,
            origin_channel,
            origin_thread,
            f"*Reviewer guidance:* {text}\n_(settled after {approval['turns']} rounds of review)_",
        )
        resolve_approval(conn, approval_id=approval["id"], guidance=text)
        return

    # Guidance becomes context the agent acts on: recorded in the origin
    # thread, so the next run (history-aware) carries the reviewer's direction.
    store.add(
        channel_id=origin_channel,
        thread_ts=origin_thread,
        role="system",
        content=f"Reviewer direction: {text}",
    )
    placeholder = client.chat_postMessage(
        channel=origin_channel,
        thread_ts=origin_thread,
        text="Acting on reviewer guidance… :eyes:",
    )
    _note_active(origin_channel, approval["question"])

    def progress(stage: str) -> None:
        try:
            client.chat_update(
                channel=origin_channel, ts=placeholder["ts"], text=f"{stage}… :eyes:"
            )
        except Exception as error:  # noqa: BLE001 - progress is best-effort
            logger.warning("progress update failed: %s", error)

    try:
        result = agent.handle(
            question=approval["question"],
            channel_id=origin_channel,
            thread_ts=origin_thread,
            progress=progress,
            directive=text,
        )
    except Exception as error:  # noqa: BLE001 - surface failures instead of silent threads
        logger.exception("directed re-run failed: %s", error)
        _clear_active(origin_channel)
        _fail(settings, client, origin_channel, placeholder["ts"], error, origin_thread)
        return
    _clear_active(origin_channel)
    reply = _format_reply(result) + "\n_(under reviewer direction)_"

    try:
        _deliver(client, origin_channel, origin_thread, placeholder["ts"], reply)
        store.add(
            channel_id=origin_channel, thread_ts=origin_thread, role="assistant", content=reply
        )
    except Exception as error:  # noqa: BLE001 - never strand a thread silently
        logger.exception(
            "directed-reply delivery failed for approval %s: %s", approval["id"], error
        )
        _delivery_failed(client, origin_channel, placeholder["ts"])

    turns = bump_approval_turns(conn, approval_id=approval["id"])
    if not result.needs_human or turns >= MAX_REVIEW_TURNS:
        resolve_approval(conn, approval_id=approval["id"], guidance=text)


def _format_reply(result: AgentResult) -> str:
    reply = result.answer
    if result.sources and result.sources != ["clarify"]:
        reply += "\n_Sources: " + ", ".join(result.sources) + "_"
    if result.warnings:
        reply += "\n⚠️ " + " ".join(result.warnings)
    return reply


def _post(client, channel_id, thread_ts, text) -> None:
    client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=text)


def _react(client, channel_id, message_ts, name) -> None:
    if not message_ts:
        return
    try:
        client.reactions_add(channel=channel_id, timestamp=message_ts, name=name)
    except Exception as error:  # noqa: BLE001 - already-reacted / missing ts are fine
        logger.debug("react %s failed: %s", name, error)


def _unreact(client, channel_id, message_ts, name) -> None:
    if not message_ts:
        return
    try:
        client.reactions_remove(channel=channel_id, timestamp=message_ts, name=name)
    except Exception as error:  # noqa: BLE001 - best-effort cleanup
        logger.debug("unreact %s failed: %s", name, error)


def _escalate(settings, conn, client, channel_id, thread_ts, question, result) -> None:
    reason = result.escalation_reason or "low confidence"
    response = client.chat_postMessage(
        channel=settings.slack_admin_channel_id,
        text=(
            ":bust_in_silhouette: Review needed\n"
            f"*Question:* {question}\n"
            f"*Reason:* {reason}\n"
            "Reply in this thread (or in the original thread) with guidance."
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


def _chunk_text(text: str, limit: int = SLACK_TEXT_LIMIT) -> list[str]:
    """Split text into Slack-safe chunks at line boundaries, hard-splitting
    any single line longer than the limit. Never loses content."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines() or [""]:
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if not current:
            current = line
        elif len(current) + 1 + len(line) <= limit:
            current += "\n" + line
        else:
            chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return chunks


def _deliver(client, channel_id, thread_ts, placeholder_ts, reply: str) -> None:
    """Post the final reply. chat.update caps at 4,000 chars, so long replies
    update the placeholder with the first chunk and post the rest as
    thread messages (chat.postMessage allows up to 40,000)."""
    chunks = _chunk_text(reply)
    client.chat_update(channel=channel_id, ts=placeholder_ts, text=chunks[0])
    for chunk in chunks[1:]:
        client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=chunk)


def _delivery_failed(client, channel_id, placeholder_ts) -> None:
    try:
        client.chat_update(
            channel=channel_id,
            ts=placeholder_ts,
            text="The answer was generated but failed to post. Check the bot logs.",
        )
    except Exception as error:  # noqa: BLE001 - best-effort notice
        logger.warning("delivery-failure notice failed: %s", error)


def _failure_notice(error: Exception, ref: str) -> tuple[str, bool]:
    """User-facing notice + whether it looks like an infrastructure failure."""
    if isinstance(error, (psycopg.OperationalError, psycopg.InterfaceError)):
        return (
            f"I can't reach the data warehouse right now. Flagged. Retry shortly. (ref {ref})",
            True,
        )
    if isinstance(error, (anthropic.APIError, OSError)):
        return f"The model is unavailable or timed out. Retry. (ref {ref})", True
    return f"Something went wrong finishing that. Retry. (ref {ref})", False


def _fail(settings, client, channel_id, placeholder_ts, error: Exception, ref: str) -> None:
    notice, infra = _failure_notice(error, ref)
    try:
        client.chat_update(channel=channel_id, ts=placeholder_ts, text=notice)
    except Exception as update_error:  # noqa: BLE001 - best-effort notice
        logger.warning("failure notice failed: %s", update_error)
    if not (infra and settings.slack_admin_channel_id):
        return
    key = type(error).__name__
    if time.time() - _ALERTED.get(key, 0) < 600:
        return
    _ALERTED[key] = time.time()
    try:
        client.chat_postMessage(
            channel=settings.slack_admin_channel_id,
            text=f":warning: Bot error (ref {ref}): {key}: {error}",
        )
    except Exception as alert_error:  # noqa: BLE001 - best-effort alert
        logger.warning("admin alert failed: %s", alert_error)


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
    log_path = Path(settings.usage_log_path).parent / "bot.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    if not settings.socket_mode:
        raise SystemExit("HTTP transport not implemented yet. Set SOCKET_MODE=true.")
    if not settings.slack_bot_token or not settings.slack_app_token:
        raise SystemExit("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required.")

    store, conn = _build_store(settings)
    agent = None
    if conn is not None and settings.anthropic_api_key:
        agent = ResearchAgent(
            llm=create_client(settings),
            conn=conn,
            embedder=Embedder(),
            conversation=store,
            scope_max_tokens=settings.scope_max_tokens,
            sql_max_tokens=settings.sql_max_tokens,
            synth_max_tokens=settings.synth_max_tokens,
        )
    else:
        logger.warning("Agent disabled (missing DATABASE_URL or ANTHROPIC_API_KEY)")

    app = build_app(settings, store=store, agent=agent, conn=conn)
    logger.info("Starting Socket Mode handler")
    SocketModeHandler(app, settings.slack_app_token).start()


if __name__ == "__main__":
    main()
