import random
from datetime import date

from research_agent.transactions import (
    ListingFeature,
    fit_occupancy,
    predict_occupancy,
    simulate,
)


def _listings() -> list[ListingFeature]:
    return [
        ListingFeature(1, price=120.0, occupancy=0.6, is_entire_home=True, reviews=80),
        ListingFeature(2, price=60.0, occupancy=0.4, is_entire_home=False, reviews=10),
        ListingFeature(3, price=200.0, occupancy=0.7, is_entire_home=True, reviews=200),
    ]


def test_fit_occupancy_returns_finite_coefficients() -> None:
    coefficients = fit_occupancy(_listings())

    assert len(coefficients) == 4
    assert all(abs(float(c)) < 100 for c in coefficients)


def test_predict_uses_intercept() -> None:
    assert predict_occupancy([0.5, 0.0, 0.0, 0.0], _listings()[0]) == 0.5


def test_simulate_produces_invoices_and_signed_refunds() -> None:
    listings = _listings()
    coefficients = fit_occupancy(listings)

    entries = simulate(listings, coefficients, date(2026, 7, 1), 3, random.Random(1))

    assert entries, "expected a non-empty ledger"
    types = {entry[2] for entry in entries}
    assert "invoice" in types
    assert all(entry[3] > 0 for entry in entries if entry[2] == "invoice")
    assert all(entry[3] < 0 for entry in entries if entry[2] == "refund")


def test_simulate_is_reproducible() -> None:
    listings = _listings()
    coefficients = fit_occupancy(listings)

    first = simulate(listings, coefficients, date(2026, 7, 1), 3, random.Random(7))
    second = simulate(listings, coefficients, date(2026, 7, 1), 3, random.Random(7))

    assert first == second
