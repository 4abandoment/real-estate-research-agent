"""Load GPU-produced review embedding shards into pgvector.

Reads ``data/raw/review_embeddings/*.npz`` (written by the Colab script), stages
them, then joins review text to populate ``review_embeddings``.
"""

from pathlib import Path

import numpy as np

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "review_embeddings"
STAGING_COLUMNS = "review_id,listing_id,embedding"


def _literals(block: np.ndarray) -> list[str]:
    rounded = np.round(block.astype(np.float32), 6)
    return ["[" + ",".join(map(str, row)) + "]" for row in rounded.tolist()]


def main() -> None:
    shards = sorted(RAW.glob("*.npz"))
    if not shards:
        raise SystemExit(
            f"No shards in {RAW}. Run scripts/colab_embed_reviews.py in Colab, "
            "download the output, then re-run this."
        )

    database_url = load_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required.")

    conn = connect(database_url)
    apply_schema(conn)
    conn.execute("TRUNCATE review_embeddings, review_embedding_staging")

    total = 0
    for shard in shards:
        data = np.load(shard)
        ids = data["review_id"].tolist()
        listings = data["listing_id"].tolist()
        literals = _literals(data["embedding"])
        with conn.cursor().copy(
            f"COPY review_embedding_staging ({STAGING_COLUMNS}) FROM STDIN"
        ) as copy:
            for review_id, listing_id, literal in zip(ids, listings, literals, strict=True):
                copy.write_row((review_id, listing_id, literal))
        total += len(ids)
        print(f"staged {total} from {shard.name}", flush=True)

    conn.execute(
        "INSERT INTO review_embeddings (review_id, listing_id, content, embedding)"
        " SELECT s.review_id, s.listing_id, left(r.comments, 1000), s.embedding"
        " FROM review_embedding_staging s JOIN reviews r ON r.id = s.review_id"
    )
    conn.execute("TRUNCATE review_embedding_staging")

    count = conn.execute("SELECT count(*) FROM review_embeddings").fetchone()[0]
    print(f"review_embeddings loaded: {count}")


if __name__ == "__main__":
    main()
