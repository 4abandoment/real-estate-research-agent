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

CREATE TABLE IF NOT EXISTS listings (
    id BIGINT PRIMARY KEY,
    name TEXT,
    host_id BIGINT,
    host_name TEXT,
    neighbourhood TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    room_type TEXT,
    price_gbp NUMERIC(10, 2),
    minimum_nights INTEGER,
    number_of_reviews INTEGER,
    review_scores_rating NUMERIC(3, 2),
    availability_365 INTEGER,
    license TEXT
);

CREATE TABLE IF NOT EXISTS calendar (
    listing_id BIGINT NOT NULL,
    date DATE NOT NULL,
    available BOOLEAN,
    price_gbp NUMERIC(10, 2),
    adjusted_price_gbp NUMERIC(10, 2),
    minimum_nights INTEGER,
    maximum_nights INTEGER,
    PRIMARY KEY (listing_id, date)
);

CREATE TABLE IF NOT EXISTS reviews (
    id BIGINT PRIMARY KEY,
    listing_id BIGINT NOT NULL,
    date DATE,
    reviewer_id BIGINT,
    reviewer_name TEXT,
    comments TEXT
);

CREATE INDEX IF NOT EXISTS reviews_listing_idx ON reviews (listing_id);

CREATE TABLE IF NOT EXISTS land_registry (
    transaction_id TEXT PRIMARY KEY,
    price INTEGER,
    date_of_transfer DATE,
    postcode TEXT,
    property_type TEXT,
    old_new TEXT,
    duration TEXT,
    paon TEXT,
    saon TEXT,
    street TEXT,
    locality TEXT,
    town_city TEXT,
    district TEXT,
    county TEXT,
    ppd_category_type TEXT,
    record_status TEXT
);

CREATE INDEX IF NOT EXISTS land_registry_postcode_idx ON land_registry (postcode);

CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    listing_id BIGINT NOT NULL,
    period DATE NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('invoice', 'payment', 'refund', 'chargeback')),
    amount_gbp NUMERIC(12, 2) NOT NULL,
    status TEXT NOT NULL,
    due_date DATE,
    created_at DATE NOT NULL,
    reference TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS transactions_listing_idx ON transactions (listing_id);
CREATE INDEX IF NOT EXISTS transactions_status_idx ON transactions (status);
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
