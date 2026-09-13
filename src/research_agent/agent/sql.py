"""Text-to-SQL with strict read-only validation, execution and formatting."""

import logging
import re

from research_agent.agent.prompts import SQL_SYSTEM
from research_agent.llm.client import LLMClient
from research_agent.llm.model_router import model_for

logger = logging.getLogger(__name__)

FETCH_ROWS = 500
DISPLAY_ROWS = 20

READ_ONLY = re.compile(r"^(select|with)\b", re.IGNORECASE)
LEADING_COMMENTS = re.compile(r"^\s*(?:(?:--[^\n]*\n)|(?:/\*.*?\*/)\s*)*", re.DOTALL)
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|merge|"
    r"vacuum|analyze|comment|reindex|cluster|refresh|listen|notify|begin|commit|rollback)\b",
    re.IGNORECASE,
)
FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
UNTERMINATED_FENCE = re.compile(r"^```[a-zA-Z]*[ \t]*\n?", re.IGNORECASE)
PII_COLUMNS = re.compile(r"\b(reviewer_name|host_name)\b", re.IGNORECASE)


class SqlValidationError(ValueError):
    """Raised when generated SQL is not a single safe read-only query."""


def extract_sql(text: str) -> str:
    match = FENCE.search(text)
    result = match.group(1) if match else UNTERMINATED_FENCE.sub("", text)
    return result.replace("```", "").strip()


def validate_sql(sql: str, *, max_rows: int = FETCH_ROWS) -> str:
    text = LEADING_COMMENTS.sub("", sql.strip()).strip().rstrip(";").strip()
    if ";" in text:
        raise SqlValidationError("multiple statements are not allowed")
    if not READ_ONLY.match(text):
        raise SqlValidationError("only SELECT or WITH queries are allowed")
    if FORBIDDEN.search(text):
        raise SqlValidationError("query contains a forbidden keyword")
    if PII_COLUMNS.search(text):
        raise SqlValidationError(
            "personal name columns (reviewer_name, host_name) are not available"
        )
    if not re.search(r"\blimit\b", text, re.IGNORECASE):
        text += f"\nLIMIT {max_rows}"
    return text


def generate_sql(
    llm: LLMClient,
    question: str,
    history: str = "",
    feedback: str = "",
    max_tokens: int = 1500,
) -> str:
    content = f"Question: {question}\n\nPrior context:\n{history}"
    if feedback:
        content += f"\n\n{feedback}\nReturn a corrected query."
    messages = [{"role": "user", "content": content}]
    response = llm.complete(
        messages=messages,
        model=model_for("sql"),
        system=SQL_SYSTEM,
        max_tokens=max_tokens,
        task="sql",
    )
    if response.truncated:
        # Truncated SQL is malformed by construction; one retry at a doubled cap
        # beats shipping a syntax error into validation.
        logger.warning(
            "SQL generation hit max_tokens=%d (%d output); retrying at %d",
            max_tokens,
            response.output_tokens,
            max_tokens * 2,
        )
        response = llm.complete(
            messages=messages,
            model=model_for("sql"),
            system=SQL_SYSTEM,
            max_tokens=max_tokens * 2,
            task="sql",
        )
    return extract_sql(response.text)


def run_sql(
    conn, sql: str, *, fetch_rows: int = FETCH_ROWS, timeout_s: int = 30
) -> tuple[list[str], list[tuple]]:
    # ponytail: statement cap so a runaway scan fails fast instead of hanging the demo.
    conn.execute(f"SET statement_timeout = '{timeout_s}s'")
    try:
        cursor = conn.execute(sql)
        columns = [description.name for description in cursor.description]
        return columns, cursor.fetchmany(fetch_rows)
    finally:
        conn.execute("SET statement_timeout = 0")


def format_rows(columns: list[str], rows: list[tuple], display: int = DISPLAY_ROWS) -> str:
    lines = [" | ".join(columns)]
    for row in rows[:display]:
        lines.append(" | ".join("" if value is None else str(value) for value in row))
    if len(rows) > display:
        lines.append(f"(+{len(rows) - display} more rows)")
    return "\n".join(lines)
