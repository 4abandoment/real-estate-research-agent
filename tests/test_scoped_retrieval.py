from research_agent.agent.orchestrator import extract_listing_ids
from research_agent.search import sample_reviews


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self):
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((query, params))
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
    assert params == (30,)
    assert rows[0]["id"] == 1


def test_listing_id_sampling_passes_ids_as_params() -> None:
    conn = _FakeConn()

    sample_reviews(conn, listing_ids=[1, 2, 3])

    query, params = conn.queries[-1]
    assert "ANY(%s)" in query
    assert params == ([1, 2, 3], 30)
