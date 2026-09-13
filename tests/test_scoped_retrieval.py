from research_agent.agent.orchestrator import extract_listing_ids


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
