"""Load GPU-produced review embedding shards into pgvector.

Reads ``data/raw/review_embeddings/*.npz`` (written by the Colab script) and
COPYs the vectors straight into ``review_embeddings`` as ``halfvec`` using the
PostgreSQL **binary** COPY format (raw big-endian fp16, no text formatting).
Review text is not duplicated; it is joined from ``reviews`` at query time.

Resumable: re-running continues after the highest review_id already loaded.
Pass ``--reset`` to start over, or ``--shards N`` to limit shards.
"""

import struct
import sys
from pathlib import Path

import numpy as np

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "review_embeddings"
COLUMNS = "review_id,listing_id,embedding"
BATCH = 20_000
COPY_HEADER = b"PGCOPY\n\xff\r\n\x00" + struct.pack(">ii", 0, 0)
COPY_TRAILER = struct.pack(">h", -1)


def _binary_batch(ids: np.ndarray, listings: np.ndarray, embeddings: np.ndarray) -> bytes:
    """Encode rows as COPY BINARY: int16 nfields, then int32 len + payload per field."""
    n, dim = embeddings.shape
    ids_bytes = ids.astype(">i8").tobytes()
    listings_bytes = listings.astype(">i8").tobytes()
    vectors_bytes = embeddings.astype(">f2").tobytes()

    row_header = struct.pack(">h", 3)
    id_len = struct.pack(">i", 8)
    listing_len = struct.pack(">i", 8)
    vector_len = struct.pack(">i", 4 + dim * 2)
    dims = struct.pack(">hh", dim, 0)

    parts: list[bytes] = []
    for index in range(n):
        parts.append(row_header)
        parts.append(id_len)
        parts.append(ids_bytes[index * 8 : (index + 1) * 8])
        parts.append(listing_len)
        parts.append(listings_bytes[index * 8 : (index + 1) * 8])
        parts.append(vector_len)
        parts.append(dims)
        parts.append(vectors_bytes[index * dim * 2 : (index + 1) * dim * 2])
    return COPY_HEADER + b"".join(parts) + COPY_TRAILER


def main() -> None:
    args = sys.argv[1:]
    reset = "--reset" in args
    max_shards = next((int(a) for a in args if a.isdigit()), None)

    shards = sorted(RAW.glob("*.npz"))
    if max_shards is not None:
        shards = shards[:max_shards]
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
    # ponytail: advisory lock so an overlapping run can't collide on the PK.
    conn.execute("SELECT pg_advisory_lock(42)")
    conn.execute("CREATE TABLE IF NOT EXISTS review_embedding_shards (shard_name TEXT PRIMARY KEY)")
    if reset:
        conn.execute("TRUNCATE review_embeddings, review_embedding_shards")

    loaded = {row[0] for row in conn.execute("SELECT shard_name FROM review_embedding_shards")}
    total = 0

    for shard in shards:
        if shard.name in loaded:
            print(f"skip {shard.name} (already loaded)", flush=True)
            continue
        data = np.load(shard)
        ids = data["review_id"]
        listings = data["listing_id"]
        embeddings = data["embedding"]
        for start in range(0, len(ids), BATCH):
            stop = start + BATCH
            with conn.cursor().copy(
                f"COPY review_embeddings ({COLUMNS}) FROM STDIN WITH (FORMAT binary)"
            ) as copy:
                copy.write(
                    _binary_batch(ids[start:stop], listings[start:stop], embeddings[start:stop])
                )
        conn.execute("INSERT INTO review_embedding_shards (shard_name) VALUES (%s)", (shard.name,))
        total += len(ids)
        print(f"loaded {total} ({shard.name})", flush=True)

    count = conn.execute("SELECT count(*) FROM review_embeddings").fetchone()[0]
    print(f"review_embeddings loaded: {count}")


if __name__ == "__main__":
    main()
