"""Semantic search over review text and policy documents (pgvector)."""

import sys

from research_agent.agent.sql import strip_limit
from research_agent.config import load_settings
from research_agent.db import TOPIC_COLUMNS, apply_schema, connect
from research_agent.embeddings import Embedder


def _query_vector(embedder: Embedder, query: str) -> str:
    return Embedder.to_pgvector(embedder.embed([query])[0])


def _cohort_key_column(conn, embedded_cohort: str) -> str:
    """Name of the cohort's listing-id column (generated SQL aliases it inconsistently)."""
    cursor = conn.execute(f"WITH cohort AS ({embedded_cohort}) SELECT * FROM cohort LIMIT 0")
    names = [column.name.lower() for column in (cursor.description or [])]
    if "listing_id" in names:
        return "listing_id"
    if "id" in names:
        return "id"
    raise ValueError(f"cohort query exposes no listing id column: {names}")


def sample_reviews(
    conn,
    *,
    listing_ids: list[int] | None = None,
    cohort_sql: str | None = None,
    sample_size: int = 30,
    seed: float | None = None,
) -> list[dict]:
    """Unbiased random sample of reviews from the scoped listings.

    Pass ``cohort_sql`` to draw from every listing the cohort query matches
    (its display LIMIT is stripped first, so the sample spans the full cohort,
    not just the fetched page); pass ``listing_ids`` to scope to an explicit
    id list instead. ``seed`` makes the draw reproducible (used by the eval
    harness).
    """
    if seed is not None:
        conn.execute("SELECT setseed(%s)", (seed,))
    if cohort_sql is not None:
        # Literal % in the cohort SQL (e.g. ILIKE '%Soho%') must survive
        # psycopg's placeholder parsing of the combined query.
        embedded = strip_limit(cohort_sql).replace("%", "%%")
        key = _cohort_key_column(conn, embedded)
        rows = conn.execute(
            "WITH cohort AS ("
            + embedded
            + ") SELECT r.id, r.listing_id, r.date, r.comments FROM reviews r"
            " JOIN cohort c ON c." + key + " = r.listing_id"
            " WHERE r.comments IS NOT NULL"
            " ORDER BY random() LIMIT %s",
            (sample_size,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT r.id, r.listing_id, r.date, r.comments FROM reviews r"
            " WHERE r.listing_id = ANY(%s) AND r.comments IS NOT NULL"
            " ORDER BY random() LIMIT %s",
            (listing_ids or [], sample_size),
        ).fetchall()
    return [{"id": row[0], "listing_id": row[1], "date": row[2], "content": row[3]} for row in rows]


def cohort_review_stats(conn, *, cohort_sql: str) -> str | None:
    """Cohort-wide sentiment and topic rates, with the portfolio baseline.

    Runs over every enriched review of the cohort (not the sampled page), so
    answers can quote real statistics next to the qualitative sample.
    """
    embedded = strip_limit(cohort_sql).replace("%", "%%")
    key = _cohort_key_column(conn, embedded)
    topic_selects = ", ".join(f"count(*) FILTER (WHERE r.topic_{topic})" for topic in TOPIC_COLUMNS)
    row = conn.execute(
        "WITH cohort AS (" + embedded + ") SELECT count(*),"
        " count(*) FILTER (WHERE r.sentiment IS NOT NULL),"
        " count(*) FILTER (WHERE r.sentiment = 'positive'),"
        " count(*) FILTER (WHERE r.sentiment = 'neutral'),"
        " count(*) FILTER (WHERE r.sentiment = 'negative'), "
        + topic_selects
        + " FROM reviews r JOIN cohort c ON c."
        + key
        + " = r.listing_id",
    ).fetchone()
    total = int(row[0])
    enriched = int(row[1])
    if enriched == 0:
        return None
    positive, neutral, negative = int(row[2]), int(row[3]), int(row[4])
    topic_counts = {topic: int(row[5 + index]) for index, topic in enumerate(TOPIC_COLUMNS)}
    baseline = conn.execute(
        "SELECT total_reviews, positive_pct, neutral_pct, negative_pct, "
        + ", ".join(f"topic_{topic}_pct" for topic in TOPIC_COLUMNS)
        + " FROM review_baseline WHERE id = 1"
    ).fetchone()

    def share(count: int) -> float:
        return 100.0 * count / enriched

    lines = ["Cohort review statistics (SQL over every enriched review in the cohort):"]
    if enriched < total:
        lines.append(f"- reviews in cohort: {enriched:,} enriched of {total:,} total")
    else:
        lines.append(f"- reviews in cohort: {enriched:,}")
    if baseline is not None and baseline[0]:
        lines.append(
            f"- sentiment: {share(positive):.0f}% positive / {share(neutral):.0f}% neutral"
            f" / {share(negative):.0f}% negative (portfolio average: {baseline[1]:.0f}% /"
            f" {baseline[2]:.0f}% / {baseline[3]:.0f}%)"
        )
        rates = ", ".join(
            f"{topic} {share(topic_counts[topic]):.0f}% (portfolio {baseline[4 + index]:.0f}%)"
            for index, topic in enumerate(TOPIC_COLUMNS)
        )
    else:
        lines.append(
            f"- sentiment: {share(positive):.0f}% positive / {share(neutral):.0f}% neutral"
            f" / {share(negative):.0f}% negative"
        )
        rates = ", ".join(f"{topic} {share(topic_counts[topic]):.0f}%" for topic in TOPIC_COLUMNS)
    lines.append("- topic rates: " + rates)
    return "\n".join(lines)


def search_reviews(
    conn, embedder: Embedder, query: str, top_k: int = 5, listing_ids: list[int] | None = None
) -> list[dict]:
    vector = _query_vector(embedder, query)
    if listing_ids:
        rows = conn.execute(
            "SELECT r.id, r.listing_id, r.comments, e.embedding <=> %s::halfvec AS distance"
            " FROM review_embeddings e JOIN reviews r ON r.id = e.review_id"
            " WHERE e.listing_id = ANY(%s)"
            " ORDER BY distance LIMIT %s",
            (vector, listing_ids, top_k),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT r.id, r.listing_id, r.comments, e.embedding <=> %s::halfvec AS distance"
            " FROM review_embeddings e JOIN reviews r ON r.id = e.review_id"
            " ORDER BY distance LIMIT %s",
            (vector, top_k),
        ).fetchall()
    return [
        {
            "id": row[0],
            "listing_id": row[1],
            "content": row[2],
            "distance": float(row[3]),
        }
        for row in rows
    ]


def search_policy(conn, embedder: Embedder, query: str, top_k: int = 5) -> list[dict]:
    rows = conn.execute(
        "SELECT source_url, title, content, embedding <=> %s::vector AS distance"
        " FROM policy_documents ORDER BY distance LIMIT %s",
        (_query_vector(embedder, query), top_k),
    ).fetchall()
    return [
        {
            "source_url": row[0],
            "title": row[1],
            "content": row[2],
            "distance": float(row[3]),
        }
        for row in rows
    ]


def embed_reviews(
    conn,
    embedder: Embedder,
    limit: int | None = None,
    batch_size: int = 64,
    chunk_size: int = 2000,
    reset: bool = False,
    max_chars: int = 1000,
) -> int:
    """Embed review text into pgvector.

    Keyset-paginated and resumable: re-running continues after the highest
    review_id already embedded (unless ``reset``), so a multi-hour CPU job can
    be interrupted safely. Text is truncated (``max_chars``) because a few
    reviews are huge and would otherwise exhaust ONNX memory.
    """
    if reset:
        conn.execute("TRUNCATE review_embeddings")

    last_id = conn.execute("SELECT COALESCE(max(review_id), 0) FROM review_embeddings").fetchone()[
        0
    ]
    total = 0

    while True:
        remaining = None if limit is None else limit - total
        if remaining is not None and remaining <= 0:
            break
        fetch = chunk_size if remaining is None else min(chunk_size, remaining)

        rows = conn.execute(
            "SELECT id, listing_id, left(comments, %s) FROM reviews"
            " WHERE comments IS NOT NULL AND length(comments) > 40 AND id > %s"
            " ORDER BY id LIMIT %s",
            (max_chars, last_id, fetch),
        ).fetchall()
        if not rows:
            break

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = embedder.embed([row[2] for row in batch], batch_size=batch_size)
            with conn.cursor().copy(
                "COPY review_embeddings (review_id, listing_id, embedding) FROM STDIN"
            ) as copy:
                for (review_id, listing_id, _), vector in zip(batch, vectors, strict=True):
                    copy.write_row((review_id, listing_id, Embedder.to_pgvector(vector)))

        total += len(rows)
        last_id = rows[-1][0]
        print(f"embedded {total} reviews (last id {last_id})", flush=True)

    return total


def main() -> None:
    args = sys.argv[1:]
    reset = "--reset" in args
    limit = next((int(arg) for arg in args if arg.isdigit()), None)

    database_url = load_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required.")
    conn = connect(database_url)
    apply_schema(conn)
    embedder = Embedder()

    count = embed_reviews(conn, embedder, limit=limit, reset=reset)
    print(f"reviews embedded: {count}")

    for query in ["guests complained about noise and dirt", "check-in was difficult"]:
        print(f"\nquery: {query}")
        for hit in search_reviews(conn, embedder, query, top_k=3):
            print(f"  listing {hit['listing_id']} distance={hit['distance']:.4f}")


if __name__ == "__main__":
    main()
