"""Ingest real, openly-licensed datasets into the warehouse tables.

Sources:
- Inside Airbnb (London snapshot)  - CC BY 4.0
- HM Land Registry Price Paid Data - Open Government Licence v3.0

Raw downloads land in gitignored ``data/raw/``. A small, PII-redacted sample is
written to ``data/seed/`` for tests. Runs are idempotent: tables are truncated
then reloaded.
"""

import csv
import gzip
import re
import sys
import urllib.request
from pathlib import Path

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect
from research_agent.safety.pii import redact

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SEED = ROOT / "data" / "seed"

AIRBNB = "https://data.insideairbnb.com/united-kingdom/england/london/2026-06-19/data"
PPD = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-monthly-update-new-version.csv"

LONDON_POSTCODE = re.compile(r"^(E|EC|N|NW|SE|SW|W|WC)\d")
DEFAULT_LISTING_SAMPLE = 1000

LISTING_COLS = (
    "id,name,host_id,host_name,neighbourhood,latitude,longitude,room_type,"
    "price_gbp,minimum_nights,number_of_reviews,review_scores_rating,"
    "availability_365,license"
)
CALENDAR_COLS = (
    "listing_id,date,available,price_gbp,adjusted_price_gbp,minimum_nights,maximum_nights"
)
REVIEW_COLS = "id,listing_id,date,reviewer_id,reviewer_name,comments"
LAND_REGISTRY_COLS = (
    "transaction_id,price,date_of_transfer,postcode,property_type,old_new,duration,"
    "paon,saon,street,locality,town_city,district,county,ppd_category_type,record_status"
)


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"cached {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dest.name} ...")
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    print(f"saved {dest.name} ({dest.stat().st_size // (1 << 20)} MB)")
    return dest


def number(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().replace("$", "").replace(",", "")
    if not cleaned or cleaned.lower() == "nan":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def whole(value: str | None) -> int | None:
    parsed = number(value)
    return int(parsed) if parsed is not None else None


def flag(value: str | None) -> bool | None:
    if value is None:
        return None
    marker = value.strip().lower()
    if marker in ("t", "true", "1"):
        return True
    if marker in ("f", "false", "0"):
        return False
    return None


def as_date(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip()[:10] or None


def load_listings(conn, path: Path, limit: int) -> list[int]:
    ids: list[int] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        with conn.cursor().copy(f"COPY listings ({LISTING_COLS}) FROM STDIN") as copy:
            for index, row in enumerate(reader):
                if index >= limit:
                    break
                try:
                    listing_id = int(row["id"])
                except (TypeError, ValueError):
                    continue
                copy.write_row(
                    (
                        listing_id,
                        row.get("name"),
                        whole(row.get("host_id")),
                        row.get("host_name"),
                        row.get("neighbourhood_cleansed") or row.get("neighbourhood"),
                        number(row.get("latitude")),
                        number(row.get("longitude")),
                        row.get("room_type"),
                        number(row.get("price")),
                        whole(row.get("minimum_nights")),
                        whole(row.get("number_of_reviews")),
                        number(row.get("review_scores_rating")),
                        whole(row.get("availability_365")),
                        row.get("license"),
                    )
                )
                ids.append(listing_id)
    return ids


def load_calendar(conn, path: Path, listing_ids: list[int]) -> int:
    wanted = set(listing_ids)
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        with conn.cursor().copy(f"COPY calendar ({CALENDAR_COLS}) FROM STDIN") as copy:
            for row in reader:
                try:
                    listing_id = int(row["listing_id"])
                except (TypeError, ValueError):
                    continue
                if listing_id not in wanted:
                    continue
                copy.write_row(
                    (
                        listing_id,
                        as_date(row.get("date")),
                        flag(row.get("available")),
                        number(row.get("price")),
                        number(row.get("adjusted_price")),
                        whole(row.get("minimum_nights")),
                        whole(row.get("maximum_nights")),
                    )
                )
                count += 1
    return count


def load_reviews(conn, path: Path, listing_ids: list[int]) -> int:
    wanted = set(listing_ids)
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        with conn.cursor().copy(f"COPY reviews ({REVIEW_COLS}) FROM STDIN") as copy:
            for row in reader:
                try:
                    listing_id = int(row["listing_id"])
                    review_id = int(row["id"])
                except (TypeError, ValueError):
                    continue
                if listing_id not in wanted:
                    continue
                copy.write_row(
                    (
                        review_id,
                        listing_id,
                        as_date(row.get("date")),
                        whole(row.get("reviewer_id")),
                        row.get("reviewer_name"),
                        row.get("comments"),
                    )
                )
                count += 1
    return count


def load_land_registry(conn, path: Path, postcode_regex: re.Pattern = LONDON_POSTCODE) -> int:
    count = 0
    with path.open("rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        with conn.cursor().copy(f"COPY land_registry ({LAND_REGISTRY_COLS}) FROM STDIN") as copy:
            for row in reader:
                if len(row) < 16:
                    continue
                postcode = row[3].strip()
                if not postcode_regex.match(postcode):
                    continue
                copy.write_row(
                    (
                        row[0],
                        whole(row[1]),
                        as_date(row[2]),
                        postcode,
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        row[9],
                        row[10],
                        row[11],
                        row[12],
                        row[13],
                        row[14],
                        row[15],
                    )
                )
                count += 1
    return count


def _write_csv(path: Path, headers: list[str], rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_seed(conn, rows: int = 20) -> None:
    """Write a small, PII-redacted sample for tests (safe to commit)."""
    SEED.mkdir(parents=True, exist_ok=True)

    listings = conn.execute(
        "SELECT id, name, neighbourhood, room_type, price_gbp, minimum_nights"
        " FROM listings ORDER BY id LIMIT %s",
        (rows,),
    ).fetchall()
    _write_csv(
        SEED / "listings_sample.csv",
        ["id", "name", "neighbourhood", "room_type", "price_gbp", "minimum_nights"],
        [(r[0], redact(r[1]), r[2], r[3], r[4], r[5]) for r in listings],
    )

    reviews = conn.execute(
        "SELECT id, listing_id, date, reviewer_name, comments FROM reviews ORDER BY id LIMIT %s",
        (rows,),
    ).fetchall()
    _write_csv(
        SEED / "reviews_sample.csv",
        ["id", "listing_id", "date", "reviewer_name", "comments"],
        [(r[0], r[1], r[2], "[REDACTED]", redact(r[4])) for r in reviews],
    )

    registry = conn.execute(
        "SELECT transaction_id, price, date_of_transfer, postcode, property_type, town_city"
        " FROM land_registry LIMIT %s",
        (rows,),
    ).fetchall()
    _write_csv(
        SEED / "land_registry_sample.csv",
        ["transaction_id", "price", "date_of_transfer", "postcode", "property_type", "town_city"],
        registry,
    )


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LISTING_SAMPLE
    database_url = load_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required.")

    conn = connect(database_url)
    apply_schema(conn)

    listings_path = download(f"{AIRBNB}/listings.csv.gz", RAW / "listings.csv.gz")
    calendar_path = download(f"{AIRBNB}/calendar.csv.gz", RAW / "calendar.csv.gz")
    reviews_path = download(f"{AIRBNB}/reviews.csv.gz", RAW / "reviews.csv.gz")
    ppd_path = download(PPD, RAW / "ppd-monthly.csv")

    # ponytail: truncate-and-reload is idempotent and simple; add incremental
    # upserts only if ingestion volume or freshness demands it.
    conn.execute("TRUNCATE listings, calendar, reviews, land_registry")

    listing_ids = load_listings(conn, listings_path, limit)
    print(f"listings: {len(listing_ids)}")
    print(f"calendar rows: {load_calendar(conn, calendar_path, listing_ids)}")
    print(f"review rows: {load_reviews(conn, reviews_path, listing_ids)}")
    print(f"land registry rows: {load_land_registry(conn, ppd_path)}")

    write_seed(conn)
    print(f"seed sample written to {SEED}")


if __name__ == "__main__":
    main()
