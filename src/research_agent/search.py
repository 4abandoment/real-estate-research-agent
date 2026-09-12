"""Semantic search over review text and policy documents (pgvector)."""

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect
from research_agent.embeddings import Embedder


def _query_vector(embedder: Embedder, query: str) -> str:
    return Embedder.to_pgvector(embedder.embed([query])[0])


def search_reviews(conn, embedder: Embedder, query: str, top_k: int = 5) -> list[dict]:
    rows = conn.execute(
        "SELECT listing_id, content, embedding <=> %s::vector AS distance"
        " FROM review_embeddings ORDER BY distance LIMIT %s",
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


def embed_reviews(conn, embedder: Embedder, limit: int = 50_000, batch_size: int = 256) -> int:
    rows = conn.execute(
        "SELECT id, listing_id, comments FROM reviews"
        " WHERE comments IS NOT NULL AND length(comments) > 40"
        " ORDER BY id LIMIT %s",
        (limit,),
    ).fetchall()
    conn.execute("TRUNCATE review_embeddings")
    with conn.cursor() as cursor:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = embedder.embed([row[2] for row in batch])
            for (review_id, listing_id, content), vector in zip(batch, vectors, strict=True):
                cursor.execute(
                    "INSERT INTO review_embeddings (review_id, listing_id, content, embedding)"
                    " VALUES (%s, %s, %s, %s)",
                    (review_id, listing_id, content, Embedder.to_pgvector(vector)),
                )
    return len(rows)


def main() -> None:
    database_url = load_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required.")
    conn = connect(database_url)
    apply_schema(conn)
    embedder = Embedder()

    count = embed_reviews(conn, embedder)
    print(f"reviews embedded: {count}")

    for query in ["guests complained about noise and dirt", "check-in was difficult"]:
        print(f"\nquery: {query}")
        for hit in search_reviews(conn, embedder, query, top_k=3):
            print(f"  listing {hit['listing_id']} distance={hit['distance']:.4f}")


if __name__ == "__main__":
    main()
