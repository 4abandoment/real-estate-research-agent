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
    type TEXT NOT NULL CHECK (
        type IN ('invoice', 'fee', 'tax', 'payment', 'refund', 'chargeback')
    ),
    amount_gbp NUMERIC(12, 2) NOT NULL,
    status TEXT NOT NULL,
    due_date DATE,
    created_at DATE NOT NULL,
    reference TEXT NOT NULL
);

ALTER TABLE transactions DROP CONSTRAINT IF EXISTS transactions_type_check;
ALTER TABLE transactions ADD CONSTRAINT transactions_type_check
    CHECK (type IN ('invoice', 'fee', 'tax', 'payment', 'refund', 'chargeback'));

CREATE INDEX IF NOT EXISTS transactions_listing_idx ON transactions (listing_id);
CREATE INDEX IF NOT EXISTS transactions_status_idx ON transactions (status);

CREATE TABLE IF NOT EXISTS playbooks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    retrieval TEXT NOT NULL,
    tables TEXT,
    embedding vector(384)
);

CREATE TABLE IF NOT EXISTS neighbourhood_links (
    ppd_district TEXT NOT NULL,
    listing_neighbourhood TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    method TEXT NOT NULL,
    PRIMARY KEY (ppd_district, listing_neighbourhood)
);

CREATE TABLE IF NOT EXISTS policy_documents (
    id BIGSERIAL PRIMARY KEY,
    source_url TEXT NOT NULL,
    title TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384)
);

CREATE TABLE IF NOT EXISTS review_embeddings (
    review_id BIGINT PRIMARY KEY,
    listing_id BIGINT NOT NULL,
    embedding halfvec(384)
);

CREATE INDEX IF NOT EXISTS review_embeddings_listing_idx
    ON review_embeddings (listing_id);

CREATE TABLE IF NOT EXISTS query_log (
    id BIGSERIAL PRIMARY KEY,
    channel_id TEXT NOT NULL,
    thread_ts TEXT,
    question TEXT NOT NULL,
    sql TEXT,
    sources TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS query_log_channel_idx
    ON query_log (channel_id, created_at DESC);

CREATE TABLE IF NOT EXISTS pending_approvals (
    id BIGSERIAL PRIMARY KEY,
    origin_channel_id TEXT NOT NULL,
    origin_thread_ts TEXT NOT NULL,
    question TEXT NOT NULL,
    reason TEXT NOT NULL,
    admin_thread_ts TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW neighbourhood_market AS
SELECT n.listing_neighbourhood AS neighbourhood,
       (SELECT count(*) FROM listings l
         WHERE l.neighbourhood = n.listing_neighbourhood) AS listings,
       (SELECT round(avg(l.price_gbp), 2) FROM listings l
         WHERE l.neighbourhood = n.listing_neighbourhood) AS avg_listing_price,
       (SELECT count(*) FROM land_registry r
         WHERE r.district = n.ppd_district) AS sales,
       (SELECT round(avg(r.price), 0) FROM land_registry r
         WHERE r.district = n.ppd_district) AS avg_sale_price
FROM neighbourhood_links n;
"""


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, autocommit=True)
    # ponytail: probes=10 gives ~100ms lookups over 2M halfvec vectors
    # (vs seconds for exact scan); raise probes if recall matters more than speed.
    conn.execute("SET ivfflat.probes = 10")
    return conn


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


def record_query(
    conn: psycopg.Connection,
    *,
    channel_id: str,
    thread_ts: str | None,
    question: str,
    sql: str | None = None,
    sources: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO query_log (channel_id, thread_ts, question, sql, sources)"
        " VALUES (%s, %s, %s, %s, %s)",
        (channel_id, thread_ts, question, sql, sources),
    )


def last_query(conn: psycopg.Connection, *, channel_id: str) -> dict | None:
    row = conn.execute(
        "SELECT question, sql, sources, created_at FROM query_log"
        " WHERE channel_id = %s ORDER BY created_at DESC LIMIT 1",
        (channel_id,),
    ).fetchone()
    if row is None:
        return None
    return {"question": row[0], "sql": row[1], "sources": row[2], "created_at": row[3]}


def create_approval(
    conn: psycopg.Connection,
    *,
    origin_channel_id: str,
    origin_thread_ts: str,
    question: str,
    reason: str,
    admin_thread_ts: str,
) -> None:
    conn.execute(
        "INSERT INTO pending_approvals"
        " (origin_channel_id, origin_thread_ts, question, reason, admin_thread_ts)"
        " VALUES (%s, %s, %s, %s, %s)",
        (origin_channel_id, origin_thread_ts, question, reason, admin_thread_ts),
    )


def find_approval(conn: psycopg.Connection, *, admin_thread_ts: str) -> dict | None:
    row = conn.execute(
        "SELECT origin_channel_id, origin_thread_ts, question FROM pending_approvals"
        " WHERE admin_thread_ts = %s AND status = 'pending'"
        " ORDER BY created_at DESC LIMIT 1",
        (admin_thread_ts,),
    ).fetchone()
    if row is None:
        return None
    return {"origin_channel_id": row[0], "origin_thread_ts": row[1], "question": row[2]}


def resolve_approval(conn: psycopg.Connection, *, admin_thread_ts: str, guidance: str) -> None:
    conn.execute(
        "UPDATE pending_approvals SET status = 'resolved', reason = %s"
        " WHERE admin_thread_ts = %s AND status = 'pending'",
        (guidance, admin_thread_ts),
    )
