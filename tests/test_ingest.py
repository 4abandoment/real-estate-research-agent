from research_agent.ingest import as_date, flag, number, whole


def test_number_strips_currency_and_commas() -> None:
    assert number("$1,234.50") == 1234.5


def test_number_handles_blank_and_nan() -> None:
    assert number("") is None
    assert number("NaN") is None
    assert number(None) is None


def test_whole_casts_to_int() -> None:
    assert whole("42") == 42


def test_flag_parses_airbnb_booleans() -> None:
    assert flag("t") is True
    assert flag("f") is False
    assert flag("") is None


def test_as_date_truncates_timestamp() -> None:
    assert as_date("2026-07-01 00:00") == "2026-07-01"
    assert as_date(None) is None
