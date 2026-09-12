"""Source registry (playbooks) and embedding-based source prioritisation.

Each data source has a small YAML description. We embed the description and
store it in pgvector, so a user question can be matched to the most relevant
sources by cosine similarity instead of brittle keyword rules.
"""

from pathlib import Path

import yaml

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect
from research_agent.embeddings import Embedder

PLAYBOOK_DIR = Path(__file__).resolve().parents[2] / "data" / "playbooks"
REQUIRED_KEYS = {"id", "name", "description", "retrieval"}


def load_playbooks(directory: Path = PLAYBOOK_DIR) -> list[dict]:
    playbooks = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = REQUIRED_KEYS - data.keys()
        if missing:
            raise ValueError(f"{path.name} missing keys: {sorted(missing)}")
        playbooks.append(data)
    return playbooks


def embed_text(playbook: dict) -> str:
    return f"{playbook['name']}. {playbook['description']}"


def seed_playbooks(conn, embedder: Embedder, directory: Path = PLAYBOOK_DIR) -> int:
    playbooks = load_playbooks(directory)
    vectors = embedder.embed([embed_text(item) for item in playbooks])
    conn.execute("TRUNCATE playbooks")
    with conn.cursor() as cursor:
        for playbook, vector in zip(playbooks, vectors, strict=True):
            cursor.execute(
                "INSERT INTO playbooks (id, name, description, retrieval, tables, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    playbook["id"],
                    playbook["name"],
                    playbook["description"],
                    playbook["retrieval"],
                    ",".join(playbook.get("tables", [])),
                    Embedder.to_pgvector(vector),
                ),
            )
    return len(playbooks)


def match_sources(conn, embedder: Embedder, query: str, top_k: int = 3) -> list[dict]:
    vector = Embedder.to_pgvector(embedder.embed([query])[0])
    rows = conn.execute(
        "SELECT id, name, retrieval, tables, embedding <=> %s::vector AS distance"
        " FROM playbooks ORDER BY distance LIMIT %s",
        (vector, top_k),
    ).fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "retrieval": row[2],
            "tables": row[3],
            "distance": float(row[4]),
        }
        for row in rows
    ]


def main() -> None:
    database_url = load_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required.")

    conn = connect(database_url)
    apply_schema(conn)
    embedder = Embedder()
    count = seed_playbooks(conn, embedder)
    print(f"playbooks seeded: {count}")

    for query in [
        "What is our overdue rent across the portfolio?",
        "What did guests complain about in their reviews?",
        "How do recent sale prices compare in Camden?",
    ]:
        print(f"\nquery: {query}")
        for match in match_sources(conn, embedder, query):
            print(f"  {match['id']:<22} distance={match['distance']:.4f}")


if __name__ == "__main__":
    main()
