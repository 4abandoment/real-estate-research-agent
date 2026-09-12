"""Verify GPU (sentence-transformers) and local (fastembed) embeddings agree.

Bulk embeddings are produced on a Colab GPU while queries are embedded locally.
If pooling or normalisation differed, retrieval would silently degrade, so this
compares the two for the same review text.
"""

from pathlib import Path

import numpy as np

from research_agent.config import load_settings
from research_agent.db import connect
from research_agent.embeddings import Embedder

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "review_embeddings"
SAMPLE = 50
THRESHOLD = 0.98


def cosine(left, right) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    shards = sorted(RAW.glob("*.npz"))
    if not shards:
        raise SystemExit(f"No shards in {RAW}.")

    data = np.load(shards[0])
    ids = data["review_id"][:SAMPLE].tolist()
    embeddings = data["embedding"][:SAMPLE]

    conn = connect(load_settings().database_url)
    rows = conn.execute(
        "SELECT id, left(comments, 1000) FROM reviews WHERE id = ANY(%s)", (ids,)
    ).fetchall()
    texts = {row[0]: row[1] for row in rows}

    embedder = Embedder()
    similarities = []
    for review_id, gpu_vector in zip(ids, embeddings, strict=True):
        text = texts.get(review_id)
        if not text:
            continue
        local_vector = embedder.embed([text])[0]
        similarities.append(cosine(local_vector, gpu_vector))

    if not similarities:
        raise SystemExit("No overlapping review ids to compare.")

    mean = float(np.mean(similarities))
    print(f"compared {len(similarities)} pairs; mean cosine={mean:.4f}")

    if mean < THRESHOLD:
        raise SystemExit("INCOMPATIBLE: GPU and local embeddings disagree.")
    print("compatible")


if __name__ == "__main__":
    main()
