"""The research agent: scope -> route -> retrieve -> synthesise -> escalate."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import psycopg

from research_agent.agent.parsing import parse_decision, parse_scope
from research_agent.agent.prompts import SCHEMA_CONTEXT, SCOPE_SYSTEM, SYNTH_SYSTEM
from research_agent.agent.router import SQL_SOURCES, route_question
from research_agent.agent.sql import (
    SqlValidationError,
    format_rows,
    generate_sql,
    run_sql,
    validate_sql,
)
from research_agent.db import WAREHOUSE_TABLES, record_query
from research_agent.embeddings import Embedder
from research_agent.llm.client import LLMClient
from research_agent.llm.model_router import model_for
from research_agent.safety.pii import redact
from research_agent.search import sample_reviews, search_policy, search_reviews

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass(frozen=True)
class AgentResult:
    answer: str
    sql: str | None = None
    sources: list[str] = field(default_factory=list)
    needs_human: bool = False
    escalation_reason: str | None = None


def extract_listing_ids(columns: list[str], rows: list[tuple], cap: int = 200) -> list[int]:
    """Collect distinct listing_id values from a SQL result, in row order."""
    names = [column.lower() for column in columns]
    if "listing_id" not in names:
        return []
    index = names.index("listing_id")
    ids: list[int] = []
    for row in rows:
        value = row[index]
        if value is not None and value not in ids:
            ids.append(value)
    return ids[:cap]


class ResearchAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        conn,
        embedder: Embedder,
        conversation,
        sample_size: int = 30,
        sample_seed: float | None = None,
    ) -> None:
        self._llm = llm
        self._conn = conn
        self._embedder = embedder
        self._conversation = conversation
        self._sample_size = sample_size
        self._sample_seed = sample_seed
        self._last_sql: str | None = None

    def handle(
        self,
        *,
        question: str,
        channel_id: str,
        thread_ts: str,
        progress: Progress | None = None,
    ) -> AgentResult:
        def report(stage: str) -> None:
            if progress is not None:
                progress(stage)

        report("Scoping your question")
        safe_question = redact(question)
        history = self._conversation.history(channel_id=channel_id, thread_ts=thread_ts, limit=10)
        history_text = "\n".join(
            f"{message.role}: {redact(message.content)}" for message in history
        )

        scope = self._scope(safe_question, history_text, clarified=len(history) >= 8)
        if scope["needs_clarification"]:
            answer = "Before I dig in:\n" + "\n".join(f"- {item}" for item in scope["questions"])
            record_query(
                self._conn,
                channel_id=channel_id,
                thread_ts=thread_ts,
                question=safe_question,
                sources="clarify",
            )
            return AgentResult(answer=answer, sources=["clarify"])

        report("Prioritising sources")
        matches = route_question(self._conn, self._embedder, safe_question, top_k=3)
        source_ids = [match["id"] for match in matches]
        evidence, sql = self._retrieve(safe_question, matches, history_text, report)

        report("Writing the answer")
        decision = self._synthesise(safe_question, evidence)
        record_query(
            self._conn,
            channel_id=channel_id,
            thread_ts=thread_ts,
            question=safe_question,
            sql=sql,
            sources=",".join(source_ids),
        )
        return AgentResult(
            answer=decision["answer"],
            sql=sql,
            sources=source_ids,
            needs_human=decision["needs_human"],
            escalation_reason=decision["reason"] or None,
        )

    def _scope(self, question: str, history_text: str, clarified: bool = False) -> dict:
        content = f"Question: {question}\n\nHistory:\n{history_text}"
        if clarified:
            # ponytail: history-length cap stops endless clarification loops
            content += "\n\nThe user has already clarified this; do not ask again."
        response = self._llm.complete(
            messages=[{"role": "user", "content": content}],
            model=model_for("scope"),
            system=SCOPE_SYSTEM,
            max_tokens=300,
            task="scope",
        )
        return parse_scope(response.text)

    def _retrieve(
        self,
        question: str,
        matches: list[dict],
        history_text: str,
        report: Progress | None = None,
    ) -> tuple[str, str | None]:
        def tell(stage: str) -> None:
            if report is not None:
                report(stage)

        blocks: list[str] = []
        sql: str | None = None
        listing_ids: list[int] = []

        if any(match["id"] in SQL_SOURCES for match in matches):
            tell("Querying the warehouse")
            block, listing_ids = self._sql_block(question, history_text)
            blocks.append(block)
            sql = self._last_sql

        if any(match["id"] == "guest_reviews" for match in matches):
            tell("Sampling guest reviews")
            if listing_ids:
                # Measurement base: unbiased random sample of the scoped cohort.
                sampled = sample_reviews(
                    self._conn,
                    listing_ids=listing_ids,
                    sample_size=self._sample_size,
                    seed=self._sample_seed,
                )
                if sampled:
                    blocks.append(
                        f"Random sample of {len(sampled)} reviews from the"
                        f" {len(listing_ids)} scoped listings (measurement base):\n"
                        + "\n".join(
                            f"- #{item['id']} (listing {item['listing_id']},"
                            f" {item['date']}): {redact(item['content'])}"
                            for item in sampled
                        )
                    )
            else:
                # Illustration only: most-similar excerpts, not a random sample.
                hits = search_reviews(self._conn, self._embedder, question, top_k=5)
                if hits:
                    blocks.append(
                        "Guest reviews (illustrative most-similar excerpts,"
                        " not a random sample):\n"
                        + "\n".join(
                            f"- #{hit['id']} (listing {hit['listing_id']}):"
                            f" {redact(hit['content'])}"
                            for hit in hits
                        )
                    )

        if any(match["id"] == "policy_kb" for match in matches):
            hits = search_policy(self._conn, self._embedder, question, top_k=5)
            if hits:
                blocks.append(
                    "Policy guidance:\n"
                    + "\n".join(
                        f"- [{hit['title']}] {redact(hit['content'])} ({hit['source_url']})"
                        for hit in hits
                    )
                )

        if any(match["id"] == "data_dictionary" for match in matches):
            blocks.append(self._data_dictionary_block())

        return "\n\n".join(blocks) or "No evidence retrieved.", sql

    def _data_dictionary_block(self) -> str:
        counts = {
            name: max(int(tuples), 0)
            for name, tuples in self._conn.execute(
                "SELECT relname, reltuples FROM pg_class WHERE relname = ANY(%s)",
                (list(WAREHOUSE_TABLES),),
            ).fetchall()
        }
        lines = ["Available data (approximate row counts):"]
        lines += [f"- {table}: {counts.get(table, 0):,}" for table in WAREHOUSE_TABLES]
        return "\n".join(lines) + "\n" + SCHEMA_CONTEXT

    def _sql_block(self, question: str, history_text: str) -> tuple[str, list[int]]:
        feedback = ""
        for attempt in range(2):
            candidate = generate_sql(self._llm, question, history_text, feedback=feedback)
            try:
                validated = validate_sql(candidate)
                columns, rows = run_sql(self._conn, validated)
            except (SqlValidationError, psycopg.Error) as error:
                logger.warning("SQL attempt %d failed: %s\nSQL: %s", attempt + 1, error, candidate)
                feedback = f"Previous attempt failed with: {error}\nPrevious SQL:\n{candidate}"
                continue
            self._last_sql = validated
            return (
                f"SQL result:\n{format_rows(columns, rows)}",
                extract_listing_ids(columns, rows),
            )
        self._last_sql = None
        return "SQL attempt failed after one retry; a reviewer should provide the figure.", []

    def _synthesise(self, question: str, evidence: str) -> dict:
        response = self._llm.complete(
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nEvidence:\n{evidence}",
                }
            ],
            model=model_for("synthesize"),
            system=SYNTH_SYSTEM,
            max_tokens=2000,
            task="synthesize",
        )
        return parse_decision(response.text, fallback=response.text)
