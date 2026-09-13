from types import SimpleNamespace

import pytest

from research_agent.agent.orchestrator import extract_listing_ids
from research_agent.search import cohort_review_stats, sample_reviews


class _FakeCursor:
    def __init__(self, rows, description=None):
        self._rows = rows
        self.description = description

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, columns=("listing_id",)):
        self.queries = []
        self._columns = columns

    def execute(self, query, params=None):
        self.queries.append((query, params))
        if "LIMIT 0" in query:
            description = [SimpleNamespace(name=name) for name in self._columns]
            return _FakeCursor([], description)
        return _FakeCursor([(1, 2, "2025-01-01", "too noisy")])


def test_extracts_distinct_listing_ids_in_row_order() -> None:
    ids = extract_listing_ids(["listing_id", "name"], [(3, "a"), (1, "b"), (3, "c")])

    assert ids == [3, 1]


def test_ignores_case_and_none_values() -> None:
    ids = extract_listing_ids(["LISTING_ID"], [(5,), (None,), (6,)])

    assert ids == [5, 6]


def test_returns_empty_without_listing_id_column() -> None:
    assert extract_listing_ids(["id", "price"], [(1, 2)]) == []


def test_caps_the_id_list() -> None:
    rows = [(i,) for i in range(300)]

    assert len(extract_listing_ids(["listing_id"], rows, cap=200)) == 200


def test_cohort_sampling_escapes_percent_literals() -> None:
    conn = _FakeConn()

    rows = sample_reviews(
        conn,
        cohort_sql="SELECT id AS listing_id FROM listings WHERE name ILIKE '%Soho%' LIMIT 500",
        sample_size=30,
    )

    query, params = conn.queries[-1]
    assert "%%Soho%%" in query
    assert "LIMIT 500" not in query
    assert "JOIN cohort c ON c.listing_id = r.listing_id" in query
    assert params == (30,)
    assert rows[0]["id"] == 1


def test_cohort_sampling_uses_id_column_alias() -> None:
    conn = _FakeConn(columns=("id", "name"))

    sample_reviews(conn, cohort_sql="SELECT l.id, l.name FROM listings l LIMIT 500")

    query, _ = conn.queries[-1]
    assert "JOIN cohort c ON c.id = r.listing_id" in query


def test_cohort_without_id_column_raises() -> None:
    conn = _FakeConn(columns=("price",))

    with pytest.raises(ValueError, match="no listing id column"):
        sample_reviews(conn, cohort_sql="SELECT l.price FROM listings l")


def test_listing_id_sampling_passes_ids_as_params() -> None:
    conn = _FakeConn()

    sample_reviews(conn, listing_ids=[1, 2, 3])

    query, params = conn.queries[-1]
    assert "ANY(%s)" in query
    assert params == ([1, 2, 3], 30)


class _StatsConn:
    def __init__(self, stats_row, baseline_row):
        self._stats_row = stats_row
        self._baseline_row = baseline_row
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append(query)
        if "LIMIT 0" in query:
            return _FakeCursor([], [SimpleNamespace(name="listing_id")])
        if "review_baseline" in query:
            return _FakeCursor([self._baseline_row] if self._baseline_row else [])
        return _FakeCursor([self._stats_row] if self._stats_row else [])


def test_cohort_review_stats_formats_baseline_comparison() -> None:
    stats_row = (1000, 800, 600, 150, 50, 100, 80, 70, 60, 50, 40, 30, 20, 10, 5)
    baseline_row = (2237083, 91.0, 6.0, 3.0, *([10.0] * 10))
    conn = _StatsConn(stats_row, baseline_row)

    text = cohort_review_stats(
        conn, cohort_sql="SELECT l.id AS listing_id FROM listings l LIMIT 500"
    )

    assert "800 scoreable of 1,000 total" in text
    assert "75% positive / 19% neutral / 6% negative" in text
    assert "portfolio average: 91% / 6% / 3%" in text
    assert "noise 10% (portfolio 10%)" in text


def test_cohort_review_stats_returns_none_when_unenriched() -> None:
    conn = _StatsConn((0, 0, 0, 0, 0, *([0] * 10)), None)

    assert cohort_review_stats(conn, cohort_sql="SELECT 1 AS listing_id") is None
