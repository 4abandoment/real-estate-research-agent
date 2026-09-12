"""Generate a modelled transaction ledger from real occupancy and pricing.

No openly-licensed AP/AR dataset exists, so the ledger is synthetic but
statistically grounded: occupancy is regressed on real listing features derived
from the Airbnb calendar, perturbed with noise, then used to derive invoices,
payments and refunds. Reproducible via a fixed random seed.
"""

import calendar
import random
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect

START = date(2026, 7, 1)
PERIODS = 6
AVG_STAY_NIGHTS = 3.5
SERVICE_FEE_RATE = 0.15
VAT_RATE = 0.20
CLEANING_FEE_GBP = 45.0
SEED = 42

SEASONALITY = {
    1: 0.90,
    2: 0.90,
    3: 1.00,
    4: 1.05,
    5: 1.10,
    6: 1.15,
    7: 1.20,
    8: 1.20,
    9: 1.10,
    10: 1.00,
    11: 0.95,
    12: 1.10,
}

TRANSACTION_COLS = "listing_id,period,type,amount_gbp,status,due_date,created_at,reference"


@dataclass(frozen=True)
class ListingFeature:
    listing_id: int
    price: float
    occupancy: float
    is_entire_home: bool
    reviews: int


def add_months(anchor: date, months: int) -> date:
    total = anchor.month - 1 + months
    return date(anchor.year + total // 12, total % 12 + 1, 1)


def days_in_month(anchor: date) -> int:
    return calendar.monthrange(anchor.year, anchor.month)[1]


def _features(feature: ListingFeature) -> list[float]:
    return [
        1.0,
        float(np.log(max(feature.price, 1.0))),
        1.0 if feature.is_entire_home else 0.0,
        float(np.log1p(feature.reviews)),
    ]


def fit_occupancy(listings: list[ListingFeature]) -> np.ndarray:
    matrix = np.array([_features(item) for item in listings])
    target = np.array([item.occupancy for item in listings])
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return coefficients


def predict_occupancy(coefficients: np.ndarray, feature: ListingFeature) -> float:
    return float(np.dot(coefficients, _features(feature)))


def simulate(
    listings: list[ListingFeature],
    coefficients: np.ndarray,
    start: date,
    periods: int,
    rng: random.Random,
) -> list[tuple]:
    entries: list[tuple] = []
    for feature in listings:
        base = predict_occupancy(coefficients, feature)
        for month_index in range(periods):
            month_start = add_months(start, month_index)
            days = days_in_month(month_start)
            rate = base * SEASONALITY[month_start.month] + rng.gauss(0, 0.05)
            rate = min(max(rate, 0.0), 1.0)
            booked_nights = round(rate * days)
            bookings = max(round(booked_nights / AVG_STAY_NIGHTS), 0)
            for booking in range(bookings):
                entries.extend(_booking_entries(feature, month_start, days, booking, rng))
    return entries


def _booking_entries(
    feature: ListingFeature, month_start: date, days: int, booking: int, rng: random.Random
) -> list[tuple]:
    nights = rng.randint(2, 7)
    gross = round(feature.price * nights + CLEANING_FEE_GBP, 2)
    issued = month_start + timedelta(days=rng.randint(0, days - 1))
    due = issued + timedelta(days=14)
    reference = f"{feature.listing_id}-{issued.isoformat()}-{booking}"

    roll = rng.random()
    if roll < 0.04:
        status = "overdue"
    elif roll < 0.10:
        status = "pending"
    elif roll < 0.18:
        status = "partial"
    else:
        status = "paid"

    entries = [(feature.listing_id, month_start, "invoice", gross, status, due, issued, reference)]

    if status in ("overdue", "pending"):
        entries.extend(_maybe_refund(feature, month_start, gross, issued, reference, rng))
        return entries

    service_fee = round(gross * SERVICE_FEE_RATE, 2)
    vat = round(service_fee * VAT_RATE, 2)
    payout = round(gross - service_fee - vat, 2)
    entries.append(
        (feature.listing_id, month_start, "fee", -service_fee, "deducted", None, issued, reference)
    )
    entries.append(
        (feature.listing_id, month_start, "tax", -vat, "deducted", None, issued, reference)
    )

    if status == "partial":
        paid = round(payout * rng.uniform(0.3, 0.8), 2)
        entries.append(
            (
                feature.listing_id,
                month_start,
                "payment",
                paid,
                "partial",
                due,
                issued + timedelta(days=rng.randint(0, 10)),
                reference,
            )
        )
    else:
        entries.append(
            (
                feature.listing_id,
                month_start,
                "payment",
                payout,
                "paid",
                due,
                issued + timedelta(days=rng.randint(0, 10)),
                reference,
            )
        )
        if rng.random() < 0.02:
            chargeback = -round(gross * rng.uniform(0.5, 1.0), 2)
            entries.append(
                (
                    feature.listing_id,
                    month_start,
                    "chargeback",
                    chargeback,
                    "charged_back",
                    None,
                    issued + timedelta(days=rng.randint(5, 40)),
                    reference,
                )
            )

    entries.extend(_maybe_refund(feature, month_start, gross, issued, reference, rng))
    return entries


def _maybe_refund(
    feature: ListingFeature,
    month_start: date,
    gross: float,
    issued: date,
    reference: str,
    rng: random.Random,
) -> list[tuple]:
    if rng.random() >= 0.03:
        return []
    refund = -round(gross * rng.uniform(0.2, 1.0), 2)
    return [
        (
            feature.listing_id,
            month_start,
            "refund",
            refund,
            "refunded",
            None,
            issued + timedelta(days=rng.randint(1, 20)),
            reference,
        )
    ]


def load_features(conn) -> list[ListingFeature]:
    rows = conn.execute(
        """
        SELECT l.id, l.price_gbp, l.room_type, l.number_of_reviews,
               avg(CASE WHEN c.available THEN 0 ELSE 1 END) AS occupancy
        FROM listings l
        JOIN calendar c ON c.listing_id = l.id
        WHERE l.price_gbp IS NOT NULL
        GROUP BY l.id, l.price_gbp, l.room_type, l.number_of_reviews
        """
    ).fetchall()
    return [
        ListingFeature(
            listing_id=row[0],
            price=float(row[1]),
            occupancy=float(row[4]),
            is_entire_home=row[2] == "Entire home/apt",
            reviews=int(row[3] or 0),
        )
        for row in rows
    ]


def main() -> None:
    database_url = load_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required.")

    conn = connect(database_url)
    apply_schema(conn)

    listings = load_features(conn)
    if not listings:
        raise SystemExit("No listings found. Run `python -m research_agent.ingest` first.")

    coefficients = fit_occupancy(listings)
    rng = random.Random(SEED)
    entries = simulate(listings, coefficients, START, PERIODS, rng)

    # ponytail: truncate-and-reload keeps simulation reproducible.
    conn.execute("TRUNCATE transactions")
    with conn.cursor().copy(f"COPY transactions ({TRANSACTION_COLS}) FROM STDIN") as copy:
        for entry in entries:
            copy.write_row(entry)

    print(f"listings modelled: {len(listings)}")
    print(f"coefficients: {[round(float(c), 4) for c in coefficients]}")
    print(f"transaction entries: {len(entries)}")


if __name__ == "__main__":
    main()
