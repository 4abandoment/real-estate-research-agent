"""LLM spot-check of the review enrichment labels (sentiment + topic flags).

Samples enriched reviews, asks a cheap LLM (OpenCode Zen) to label the same
reviews independently, and reports agreement - the validation step for the
lexicon/anchor pipeline. Disagreements are printed for human review.

Usage:
    python scripts/qa_review_enrichment.py [sample_size] [batch_size]
"""

import json
import re
import sys

from research_agent.config import load_settings
from research_agent.db import TOPIC_COLUMNS, connect
from research_agent.llm.client import create_client

MODEL = "openrouter/deepseek/deepseek-v4-flash"
MAX_TEXT_CHARS = 500
SYSTEM = (
    "You label Airbnb guest reviews for a London short-let portfolio. For each "
    'review return a JSON object with "id" (the given id), "sentiment" '
    '(positive | neutral | negative), and "topics" (array of zero or more topic '
    "ids from this fixed list: " + ", ".join(TOPIC_COLUMNS) + "). "
    "Include a topic only when the review actually mentions it. "
    "Return a JSON array only, no prose."
)
FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def parse_labels(text: str) -> list[dict]:
    match = FENCE.search(text)
    payload = match.group(1) if match else text
    return json.loads(payload.strip())


def main() -> None:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    settings = load_settings()
    conn = connect(settings.database_url)
    client = create_client(settings)

    topic_fields = ", ".join(f"topic_{topic}" for topic in TOPIC_COLUMNS)
    rows = conn.execute(
        f"SELECT id, comments, sentiment, {topic_fields} FROM reviews"
        " WHERE sentiment IS NOT NULL AND length(comments) > 40"
        " ORDER BY random() LIMIT %s",
        (size,),
    ).fetchall()

    expected: dict[int, dict] = {}
    for row in rows:
        review_id, _comments, sentiment, *flags = row
        expected[review_id] = {
            "sentiment": sentiment,
            "topics": {topic for topic, flag in zip(TOPIC_COLUMNS, flags, strict=True) if flag},
        }

    predicted: dict[int, dict] = {}
    parse_failures = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        payload = [{"id": row[0], "text": (row[1] or "")[:MAX_TEXT_CHARS]} for row in batch]
        response = client.complete(
            messages=[{"role": "user", "content": json.dumps(payload)}],
            model=MODEL,
            system=SYSTEM,
            max_tokens=4000,
            task="qa_reviews",
        )
        try:
            for item in parse_labels(response.text):
                predicted[int(item["id"])] = {
                    "sentiment": item.get("sentiment"),
                    "topics": set(item.get("topics") or []),
                }
        except (ValueError, KeyError, TypeError) as error:
            parse_failures += 1
            print(f"batch {start // batch_size + 1}: unparseable reply ({error})")

    common = sorted(set(expected) & set(predicted))
    if not common:
        raise SystemExit("no overlapping labels - QA failed")

    sentiment_hits = sum(
        1
        for review_id in common
        if expected[review_id]["sentiment"] == predicted[review_id]["sentiment"]
    )
    topic_exact = sum(
        1 for review_id in common if expected[review_id]["topics"] == predicted[review_id]["topics"]
    )
    jaccards = []
    for review_id in common:
        left = expected[review_id]["topics"]
        right = predicted[review_id]["topics"]
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)

    print(
        f"\nsampled {len(rows)} reviews, compared {len(common)} (batches failed: {parse_failures})"
    )
    print(f"sentiment agreement: {100.0 * sentiment_hits / len(common):.1f}%")
    print(f"topic set exact match: {100.0 * topic_exact / len(common):.1f}%")
    print(f"topic overlap (mean Jaccard): {sum(jaccards) / len(jaccards):.2f}")

    print("\n-- disagreements (up to 10 for human review) --")
    shown = 0
    for review_id in common:
        exp, got = expected[review_id], predicted[review_id]
        if exp["sentiment"] == got["sentiment"] and exp["topics"] == got["topics"]:
            continue
        text = next(row[1] for row in rows if row[0] == review_id)
        print(
            f"#{review_id}: pipeline={exp['sentiment']} {sorted(exp['topics'])} | "
            f"llm={got['sentiment']} {sorted(got['topics'])}"
        )
        print(f"    {(text or '')[:160]!r}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
