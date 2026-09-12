"""Semantic search over review text and policy documents (pgvector)."""

import sys

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect
from research_agent.embeddings import Embedder


def _query_vector(embedder: Embedder, query: str) -> str:
    return Embedder.to_pgvector(embedder.embed([query])[0])


def search_reviews(conn, embedder: Embedder, query: str, top_k: int = 5) -> list[dict]:
    rows = conn.execute(
        "SELECT r.listing_id, r.comments, e.embedding <=> %s::halfvec AS distance"
        " FROM review_embeddings e JOIN reviews r ON r.id = e.review_id"
        " ORDER BY distance LIMIT %s",
        (_query_vector(embedder, query), top_k),
    ).fetchall()
    return [{"listing_id": row[0], "content": row[1], "distance": float(row[2])} for row in rows]


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
