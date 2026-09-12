import pytest

from research_agent.agent.sql import SqlValidationError, extract_sql, validate_sql


def test_extract_sql_from_fenced_block() -> None:
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"


def test_validate_adds_limit_when_missing() -> None:
    assert validate_sql("SELECT * FROM listings", max_rows=10).endswith("LIMIT 10")


def test_validate_keeps_existing_limit() -> None:
    result = validate_sql("SELECT * FROM listings LIMIT 5")
    assert result.lower().count("limit") == 1


def test_validate_allows_cte() -> None:
    result = validate_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x LIMIT 1")
    assert result.startswith("WITH")


def test_validate_strips_leading_comments() -> None:
    result = validate_sql("-- overdue rent\nSELECT 1")
    assert result.startswith("SELECT")
    assert result.endswith("LIMIT 50")


def test_validate_rejects_write_statements() -> None:
    with pytest.raises(SqlValidationError):
        validate_sql("DELETE FROM listings")


def test_validate_rejects_multiple_statements() -> None:
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT 1; DROP TABLE listings")


def test_validate_rejects_forbidden_keyword() -> None:
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT * FROM listings; CREATE TABLE x (id int)")
