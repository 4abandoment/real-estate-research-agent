"""Text-to-SQL with strict read-only validation, execution and formatting."""

import re

from research_agent.agent.prompts import SQL_SYSTEM
from research_agent.llm.client import LLMClient
from research_agent.llm.model_router import model_for

MAX_ROWS = 50

READ_ONLY = re.compile(r"^(select|with)\b", re.IGNORECASE)
LEADING_COMMENTS = re.compile(r"^\s*(?:(?:--[^\n]*\n)|(?:/\*.*?\*/)\s*)*", re.DOTALL)
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|merge|"
    r"vacuum|analyze|comment|reindex|cluster|refresh|listen|notify|begin|commit|rollback)\b",
    re.IGNORECASE,
)
FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


class SqlValidationError(ValueError):
    """Raised when generated SQL is not a single safe read-only query."""


def extract_sql(text: str) -> str:
    match = FENCE.search(text)
    return (match.group(1) if match else text).strip()


def validate_sql(sql: str, *, max_rows: int = MAX_ROWS) -> str:
    text = LEADING_COMMENTS.sub("", sql.strip()).strip().rstrip(";").strip()
    if ";" in text:
        raise SqlValidationError("multiple statements are not allowed")
    if not READ_ONLY.match(text):
        raise SqlValidationError("only SELECT or WITH queries are allowed")
    if FORBIDDEN.search(text):
        raise SqlValidationError("query contains a forbidden keyword")
    if not re.search(r"\blimit\b", text, re.IGNORECASE):
        text += f"\nLIMIT {max_rows}"
    return text


def generate_sql(llm: LLMClient, question: str, history: str = "") -> str:
    messages = [{"role": "user", "content": f"Question: {question}\n\nPrior context:\n{history}"}]
    response = llm.complete(
        messages=messages,
        model=model_for("sql"),
        system=SQL_SYSTEM,
        max_tokens=600,
        task="sql",
    )
    return extract_sql(response.text)


def run_sql(conn, sql: str, *, max_rows: int = MAX_ROWS) -> tuple[list[str], list[tuple]]:
    cursor = conn.execute(sql)
    columns = [description.name for description in cursor.description]
    return columns, cursor.fetchmany(max_rows)


def format_rows(columns: list[str], rows: list[tuple]) -> str:
    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join("" if value is None else str(value) for value in row))
    return "\n".join(lines)
