"""Postgres access, schema and conversation history."""

from dataclasses import dataclass

import psycopg

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    channel_id TEXT NOT NULL,
    thread_ts TEXT NOT NULL,
    user_id TEXT,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_thread_idx
    ON messages (channel_id, thread_ts, created_at);
"""


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, autocommit=True)


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


class ConversationStore:
    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def add(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        role: str,
        content: str,
        user_id: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO messages (channel_id, thread_ts, user_id, role, content)"
            " VALUES (%s, %s, %s, %s, %s)",
            (channel_id, thread_ts, user_id, role, content),
        )

    def history(self, *, channel_id: str, thread_ts: str, limit: int = 20) -> list[Message]:
        rows = self._conn.execute(
            "SELECT role, content FROM messages"
            " WHERE channel_id = %s AND thread_ts = %s"
            " ORDER BY created_at DESC LIMIT %s",
            (channel_id, thread_ts, limit),
        ).fetchall()
        return [Message(role=role, content=content) for role, content in reversed(rows)]
