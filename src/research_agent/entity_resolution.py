"""Resolve entities across sources.

Airbnb listings and Land Registry records share no key, so we link them at
neighbourhood level: normalise and fuzzy-match Price Paid districts to Airbnb
neighbourhoods. This is deliberately explicit about the limits of cross-source
linkage rather than pretending a false property-level join exists.
"""

import re
from difflib import SequenceMatcher

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect

PREFIXES = ("london borough of ", "royal borough of ", "borough of ", "the ")
THRESHOLD = 0.86


def normalise(name: str | None) -> str:
    text = (name or "").strip().lower()
    for prefix in PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def best_match(
    target: str, candidates: list[str], threshold: float = THRESHOLD
) -> tuple[str | None, float]:
    best: str | None = None
    best_score = 0.0
    for candidate in candidates:
        if normalise(candidate) == target:
            return candidate, 1.0
        score = SequenceMatcher(None, target, normalise(candidate)).ratio()
        if score > best_score:
            best, best_score = candidate, score
    if best_score >= threshold:
        return best, best_score
    return None, best_score


def build_neighbourhood_links(conn) -> int:
    districts = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT district FROM land_registry"
            " WHERE district IS NOT NULL AND district <> ''"
        ).fetchall()
    ]
    neighbourhoods = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT neighbourhood FROM listings"
            " WHERE neighbourhood IS NOT NULL AND neighbourhood <> ''"
        ).fetchall()
    ]

    conn.execute("TRUNCATE neighbourhood_links")
    links = 0
    with conn.cursor() as cursor:
        for district in districts:
            target = normalise(district)
            if not target:
                continue
            match, score = best_match(target, neighbourhoods)
            if match is None:
                continue
            method = "exact" if score == 1.0 else "fuzzy"
            cursor.execute(
                "INSERT INTO neighbourhood_links"
                " (ppd_district, listing_neighbourhood, score, method)"
                " VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (district, match, round(score, 3), method),
            )
            links += 1
    return links


def main() -> None:
    database_url = load_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required.")
    conn = connect(database_url)
    apply_schema(conn)
    print(f"neighbourhood links: {build_neighbourhood_links(conn)}")


if __name__ == "__main__":
    main()
