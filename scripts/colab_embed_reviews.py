"""Embed Inside Airbnb London reviews on a Google Colab GPU.

Uses the SAME model as the local query embedder (BAAI/bge-small-en-v1.5), so
the vectors are compatible with the local fastembed runtime: bulk on GPU,
queries locally, no schema change.

How to run (Colab):
  1. Runtime -> Change runtime type -> Hardware accelerator: GPU (T4).
  2. Upload this file, then run:
         !pip -q install sentence-transformers
         !python colab_embed_reviews.py
  3. Output shards land in Google Drive: MyDrive/review_embeddings/shard_*.npz
     Each shard: review_id (int64), listing_id (int64), embedding (float16, 384).
  4. Download that folder locally to data/raw/review_embeddings/ and run:
         python scripts/load_review_embeddings.py
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

REVIEWS_URL = (
    "https://data.insideairbnb.com/united-kingdom/england/london/2026-06-19/data/reviews.csv.gz"
)
OUT_DIR = "/content/drive/MyDrive/review_embeddings"
LOCAL_CSV = "/content/reviews.csv.gz"
MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHARS = 1000
SHARD = 200_000


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if not os.path.exists(LOCAL_CSV):
        print(f"downloading {REVIEWS_URL}")
        urllib.request.urlretrieve(REVIEWS_URL, LOCAL_CSV)

    frame = pd.read_csv(LOCAL_CSV, usecols=["id", "listing_id", "comments"])
    frame = frame[frame["comments"].notna()].copy()
    frame["content"] = frame["comments"].astype(str).str.slice(0, MAX_CHARS)
    frame = frame[frame["content"].str.len() > 40]
    print(f"rows: {len(frame)}")

    model = SentenceTransformer(MODEL, device=device)

    for start in range(0, len(frame), SHARD):
        shard_path = os.path.join(OUT_DIR, f"shard_{start:09d}.npz")
        if os.path.exists(shard_path):
            print(f"skip {shard_path}")
            continue
        part = frame.iloc[start : start + SHARD]
        vectors = model.encode(
            part["content"].tolist(),
            batch_size=512,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float16)
        np.savez_compressed(
            shard_path,
            review_id=part["id"].to_numpy(np.int64),
            listing_id=part["listing_id"].to_numpy(np.int64),
            embedding=vectors,
        )
        print(f"wrote {shard_path} {vectors.shape}")

    print("done")


if __name__ == "__main__":
    main()
