import random
from datetime import date

from research_agent.transactions import (
    CLEANING_FEE_GBP,
    ListingFeature,
    fit_occupancy,
    predict_occupancy,
    simulate,
)


def _listings(count: int = 40) -> list[ListingFeature]:
    return [
        ListingFeature(
            index + 1,
            price=80.0 + index * 7,
            occupancy=0.3 + (index % 5) * 0.1,
            is_entire_home=index % 2 == 0,
            reviews=index * 3,
        )
        for index in range(count)
    ]


def test_fit_occupancy_returns_finite_coefficients() -> None:
    coefficients = fit_occupancy(_listings())

    assert len(coefficients) == 4
    assert all(abs(float(c)) < 100 for c in coefficients)


def test_predict_uses_intercept() -> None:
    assert predict_occupancy([0.5, 0.0, 0.0, 0.0], _listings()[0]) == 0.5


def test_ledger_has_full_fee_model() -> None:
    listings = _listings()
    coefficients = fit_occupancy(listings)

    entries = simulate(listings, coefficients, date(2026, 7, 1), 12, random.Random(1))

    types = {entry[2] for entry in entries}
    assert {"invoice", "fee", "tax", "payment"} <= types
    assert "chargeback" in types, "2% chargeback rate should appear over 480 listings-months"

    assert all(entry[3] > 0 for entry in entries if entry[2] == "invoice")
    assert all(
        entry[3] < 0 for entry in entries if entry[2] in {"fee", "tax", "refund", "chargeback"}
    )
    assert all(entry[3] >= CLEANING_FEE_GBP for entry in entries if entry[2] == "invoice")


def test_invoices_can_be_partially_paid() -> None:
    listings = _listings(60)
    coefficients = fit_occupancy(listings)

    for seed in range(20):
        entries = simulate(listings, coefficients, date(2026, 7, 1), 6, random.Random(seed))
        statuses = {entry[4] for entry in entries if entry[2] == "invoice"}
        if "partial" in statuses:
            break
    else:
        raise AssertionError("expected at least one partial invoice")


def test_simulate_is_reproducible() -> None:
    listings = _listings()
    coefficients = fit_occupancy(listings)

    first = simulate(listings, coefficients, date(2026, 7, 1), 3, random.Random(7))
    second = simulate(listings, coefficients, date(2026, 7, 1), 3, random.Random(7))

    assert first == second
