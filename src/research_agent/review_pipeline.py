"""Guest review enrichment: VADER sentiment + embedding-anchor topic flags.

Sentiment is lexicon-based (VADER): cheap, deterministic, and validated by an
LLM spot-check rather than trusted blindly. Topics come from cosine similarity
between each review's stored embedding and LLM-authored anchor phrases
(data/review_topics.yaml) - free at 2M-row scale. BGE embeddings are
anisotropic (unrelated short texts score ~0.5), so both sides are centered on
the corpus mean before cosine; the taxonomy threshold is on that centered
scale. Reviews without embeddings still get sentiment; their topic flags stay
FALSE (short text carries little topic signal). Empty comments are scored
neutral with no topics, so every row is marked processed and the backfill
terminates.

Usage:
    python -m research_agent.review_pipeline --calibrate 2000
    python -m research_agent.review_pipeline            # full backfill, resumable
    python -m research_agent.review_pipeline --baseline # refresh baseline row
"""

import argparse
import html
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from research_agent.config import load_settings
from research_agent.db import TOPIC_COLUMNS, apply_schema, connect
from research_agent.embeddings import Embedder

logger = logging.getLogger(__name__)

TOPICS_PATH = Path(__file__).resolve().parents[2] / "data" / "review_topics.yaml"
BATCH_SIZE = 2000
CORPUS_MEAN_SAMPLE = 20000
POSITIVE_CUTOFF = 0.05
NEGATIVE_CUTOFF = -0.05
TAG_RE = re.compile(r"<[^>]+>")

ENRICH_COLUMNS = ["id", "sentiment", "sentiment_score", "pos_score", "neu_score", "neg_score"] + [
    f"topic_{topic}" for topic in TOPIC_COLUMNS
]


def load_taxonomy(path: Path = TOPICS_PATH) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = sorted(topic["id"] for topic in data["topics"])
    if ids != sorted(TOPIC_COLUMNS):
        raise ValueError(f"taxonomy ids do not match schema columns: {ids}")
    return data


def clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", raw))).strip()


def sentiment_label(compound: float) -> str:
    if compound >= POSITIVE_CUTOFF:
        return "positive"
    if compound <= NEGATIVE_CUTOFF:
        return "negative"
    return "neutral"


def parse_vector(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            return None
        return np.asarray(json.loads(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def corpus_mean(conn, sample: int = CORPUS_MEAN_SAMPLE) -> np.ndarray:
    rows = conn.execute(
        "SELECT embedding FROM review_embeddings ORDER BY random() LIMIT %s", (sample,)
    ).fetchall()
    matrix = np.asarray([parse_vector(row[0]) for row in rows], dtype=np.float32)
    return matrix.mean(axis=0)


def build_anchor_matrix(
    embedder: Embedder, taxonomy: dict, mean: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    anchors = [(topic["id"], anchor) for topic in taxonomy["topics"] for anchor in topic["anchors"]]
    matrix = np.asarray(embedder.embed([text for _, text in anchors]), dtype=np.float32)
    if mean is not None:
        matrix = matrix - mean
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    topic_index = np.asarray([TOPIC_COLUMNS.index(topic_id) for topic_id, _ in anchors])
    return matrix, topic_index


def topic_flags(
    vector: np.ndarray,
    anchors: np.ndarray,
    topic_index: np.ndarray,
    threshold: float,
    mean: np.ndarray | None = None,
) -> list[bool]:
    if mean is not None:
        vector = vector - mean
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return [False] * len(TOPIC_COLUMNS)
    sims = anchors @ (vector / norm)
    hits = sims > threshold
    flags = np.zeros(len(TOPIC_COLUMNS), dtype=bool)
    if hits.any():
        flags[np.unique(topic_index[hits])] = True
    return flags.tolist()


def enrich_rows(
    rows: list[tuple],
    analyzer: SentimentIntensityAnalyzer,
    anchors: np.ndarray,
    topic_index: np.ndarray,
    threshold: float,
    mean: np.ndarray | None = None,
) -> list[tuple]:
    results = []
    for review_id, comments, embedding in rows:
        text = clean_text(comments)
        if not text:
            empty_flags = [False] * len(TOPIC_COLUMNS)
            results.append((review_id, "neutral", 0.0, 0.0, 1.0, 0.0, *empty_flags))
            continue
        scores = analyzer.polarity_scores(text)
        vector = parse_vector(embedding)
        flags = (
            topic_flags(vector, anchors, topic_index, threshold, mean)
            if vector is not None
            else [False] * len(TOPIC_COLUMNS)
        )
        results.append(
            (
                review_id,
                sentiment_label(scores["compound"]),
                scores["compound"],
                scores["pos"],
                scores["neu"],
                scores["neg"],
                *flags,
            )
        )
    return results


def _create_batch_table(conn) -> None:
    definitions = ",\n    ".join(f"topic_{topic} BOOLEAN" for topic in TOPIC_COLUMNS)
    conn.execute(
        f"""
        CREATE TEMP TABLE IF NOT EXISTS enrich_batch (
            id BIGINT PRIMARY KEY,
            sentiment TEXT,
            sentiment_score REAL,
            pos_score REAL,
            neu_score REAL,
            neg_score REAL,
            {definitions}
        )
        """
    )


def _write_batch(conn, results: list[tuple]) -> None:
    conn.execute("TRUNCATE enrich_batch")
    with conn.cursor().copy(f"COPY enrich_batch ({', '.join(ENRICH_COLUMNS)}) FROM STDIN") as copy:
        for row in results:
            copy.write_row(row)
    assignments = ", ".join(
        [
            "sentiment = e.sentiment",
            "sentiment_score = e.sentiment_score",
            "pos_score = e.pos_score",
            "neu_score = e.neu_score",
            "neg_score = e.neg_score",
        ]
        + [f"topic_{topic} = e.topic_{topic}" for topic in TOPIC_COLUMNS]
    )
    conn.execute(f"UPDATE reviews r SET {assignments} FROM enrich_batch e WHERE r.id = e.id")


def _fetch_batch(conn, after_id: int, limit: int, *, reprocess: bool = False) -> list[tuple]:
    condition = "" if reprocess else "r.sentiment IS NULL AND "
    return conn.execute(
        """
        SELECT r.id, r.comments, e.embedding
        FROM reviews r
        LEFT JOIN review_embeddings e ON e.review_id = r.id
        WHERE """
        + condition
        + """r.id > %s
        ORDER BY r.id
        LIMIT %s
        """,
        (after_id, limit),
    ).fetchall()


def backfill(
    conn,
    embedder: Embedder,
    taxonomy: dict,
    *,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    reprocess: bool = False,
) -> int:
    analyzer = SentimentIntensityAnalyzer()
    mean = corpus_mean(conn)
    anchors, topic_index = build_anchor_matrix(embedder, taxonomy, mean)
    threshold = float(taxonomy.get("threshold", 0.3))
    _create_batch_table(conn)

    last_id = 0
    if not reprocess:
        start_row = conn.execute(
            "SELECT COALESCE(max(id), 0) FROM reviews WHERE sentiment IS NOT NULL"
        ).fetchone()
        last_id = int(start_row[0]) if start_row else 0
    done = 0
    started = time.time()
    while True:
        this_batch = batch_size if limit is None else min(batch_size, limit - done)
        if this_batch <= 0:
            break
        rows = _fetch_batch(conn, last_id, this_batch, reprocess=reprocess)
        if not rows:
            break
        results = enrich_rows(rows, analyzer, anchors, topic_index, threshold, mean)
        _write_batch(conn, results)
        conn.commit()
        done += len(rows)
        last_id = rows[-1][0]
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0.0
        print(f"enriched {done:,} reviews ({rate:,.0f}/s)", flush=True)
    return done


BASELINE_SQL_TEMPLATE = """
INSERT INTO review_baseline (
    id, built_at, total_reviews, positive_pct, neutral_pct, negative_pct, {topic_cols}
)
SELECT
    1, now(), count(*),
    100.0 * count(*) FILTER (WHERE sentiment = 'positive') / count(*),
    100.0 * count(*) FILTER (WHERE sentiment = 'neutral') / count(*),
    100.0 * count(*) FILTER (WHERE sentiment = 'negative') / count(*),
    {topic_exprs}
FROM reviews
WHERE sentiment IS NOT NULL
ON CONFLICT (id) DO UPDATE SET
    built_at = excluded.built_at,
    total_reviews = excluded.total_reviews,
    positive_pct = excluded.positive_pct,
    neutral_pct = excluded.neutral_pct,
    negative_pct = excluded.negative_pct,
    {topic_updates}
"""


def build_baseline(conn) -> dict:
    topic_cols = ", ".join(f"topic_{topic}_pct" for topic in TOPIC_COLUMNS)
    topic_exprs = ",\n    ".join(
        f"100.0 * count(*) FILTER (WHERE topic_{topic}) / count(*)" for topic in TOPIC_COLUMNS
    )
    topic_updates = ", ".join(
        f"topic_{topic}_pct = excluded.topic_{topic}_pct" for topic in TOPIC_COLUMNS
    )
    conn.execute(
        BASELINE_SQL_TEMPLATE.format(
            topic_cols=topic_cols, topic_exprs=topic_exprs, topic_updates=topic_updates
        )
    )
    conn.commit()
    row = conn.execute("SELECT * FROM review_baseline WHERE id = 1").fetchone()
    return row


def calibrate(conn, embedder: Embedder, taxonomy: dict, sample_size: int) -> None:
    mean = corpus_mean(conn)
    anchors, topic_index = build_anchor_matrix(embedder, taxonomy, mean)
    rows = conn.execute(
        """
        SELECT r.id, r.comments, e.embedding
        FROM reviews r
        JOIN review_embeddings e ON e.review_id = r.id
        WHERE length(r.comments) > 40
        ORDER BY random()
        LIMIT %s
        """,
        (sample_size,),
    ).fetchall()
    matrix = np.asarray([parse_vector(row[2]) for row in rows], dtype=np.float32) - mean
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    sims = matrix @ anchors.T
    configured = float(taxonomy.get("threshold", 0.3))
    print(f"sample: {len(rows):,} embedded reviews, {anchors.shape[0]} anchors")
    print(f"configured threshold (centered cosine): {configured:.2f}")
    for threshold in (0.25, 0.30, 0.35, 0.40, 0.45):
        hits = sims > threshold
        any_topic = np.array([np.unique(topic_index[row]).size > 0 for row in hits])
        print(f"\nthreshold {threshold:.2f}: {100.0 * any_topic.mean():.1f}% have a topic")
        for index, topic in enumerate(TOPIC_COLUMNS):
            rate = 100.0 * hits[:, topic_index == index].any(axis=1).mean()
            print(f"  {topic:<22} {rate:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process at most N reviews")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--calibrate", type=int, default=None, help="sample N reviews, print rates")
    parser.add_argument("--baseline", action="store_true", help="rebuild the baseline row only")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="re-enrich every review (use after a taxonomy/threshold change)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is required.")
    conn = connect(settings.database_url)
    apply_schema(conn)
    taxonomy = load_taxonomy()

    if args.baseline:
        row = build_baseline(conn)
        print("baseline rebuilt:", row)
        return

    embedder = Embedder()
    if args.calibrate:
        calibrate(conn, embedder, taxonomy, args.calibrate)
        return

    done = backfill(
        conn,
        embedder,
        taxonomy,
        limit=args.limit,
        batch_size=args.batch,
        reprocess=args.reprocess,
    )
    print(f"done: {done:,} reviews enriched")
    remaining = conn.execute("SELECT count(*) FROM reviews WHERE sentiment IS NULL").fetchone()[0]
    if remaining == 0 and done > 0:
        row = build_baseline(conn)
        print("baseline rebuilt:", row)
    else:
        print(f"{remaining:,} reviews still unprocessed; baseline not rebuilt")


if __name__ == "__main__":
    main()
