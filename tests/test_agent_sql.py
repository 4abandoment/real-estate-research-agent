import psycopg
import pytest

from research_agent.agent.sql import (
    SqlValidationError,
    extract_sql,
    generate_sql,
    run_sql,
    strip_limit,
    validate_sql,
)
from research_agent.llm.client import LLMResponse


class _FakeClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.messages = None

    def complete(self, *, messages, model, system=None, max_tokens=1024, task="llm"):
        self.messages = messages
        return LLMResponse(
            text=self._text, model=model, input_tokens=1, output_tokens=1, latency_ms=1.0
        )


def test_generate_sql_feeds_back_previous_error() -> None:
    client = _FakeClient("SELECT 1")
    generate_sql(client, "q", feedback="Previous attempt failed with: boom")
    assert "boom" in client.messages[0]["content"]
    assert "corrected query" in client.messages[0]["content"]


def test_strip_limit_removes_trailing_limit() -> None:
    assert strip_limit("SELECT 1 FROM t LIMIT 500") == "SELECT 1 FROM t"
    assert strip_limit("select 1 from t limit 20") == "select 1 from t"


def test_strip_limit_keeps_inner_limit() -> None:
    sql = "SELECT * FROM (SELECT 1 FROM t LIMIT 5) s"

    assert strip_limit(sql) == sql


def test_format_rows_truncates_display() -> None:
    from research_agent.agent.sql import format_rows

    rows = [(i,) for i in range(25)]
    text = format_rows(["id"], rows, display=20)

    assert "(+5 more rows)" in text
    assert text.count("\n") == 21  # header + 20 rendered rows + note


def test_run_sql_times_out_quickly() -> None:
    pytest.importorskip("research_agent.db")
    from research_agent.config import load_settings
    from research_agent.db import connect

    database_url = load_settings().database_url
    if not database_url:
        pytest.skip("DATABASE_URL not set")
    try:
        conn = connect(database_url)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")

    with pytest.raises(psycopg.errors.QueryCanceled):
        run_sql(conn, "SELECT pg_sleep(10)", timeout_s=1)


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
    assert result.endswith("LIMIT 500")


def test_validate_rejects_write_statements() -> None:
    with pytest.raises(SqlValidationError):
        validate_sql("DELETE FROM listings")


def test_validate_rejects_multiple_statements() -> None:
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT 1; DROP TABLE listings")


def test_validate_rejects_forbidden_keyword() -> None:
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT * FROM listings; CREATE TABLE x (id int)")


def test_validate_rejects_pii_name_columns() -> None:
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT reviewer_name FROM reviews LIMIT 5")
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT host_name FROM listings LIMIT 5")


def test_validate_allows_reviews_without_names() -> None:
    result = validate_sql("SELECT listing_id, date FROM reviews LIMIT 5")
    assert result.startswith("SELECT")
