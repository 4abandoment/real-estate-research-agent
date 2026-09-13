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
from research_agent.search import (
    cohort_review_stats,
    sample_reviews,
    search_policy,
    search_reviews,
)

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass(frozen=True)
class AgentResult:
    answer: str
    sql: str | None = None
    sources: list[str] = field(default_factory=list)
    needs_human: bool = False
    escalation_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


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
        scope_max_tokens: int = 500,
        sql_max_tokens: int = 3000,
        synth_max_tokens: int = 3000,
    ) -> None:
        self._llm = llm
        self._conn = conn
        self._embedder = embedder
        self._conversation = conversation
        self._sample_size = sample_size
        self._sample_seed = sample_seed
        self._scope_max_tokens = scope_max_tokens
        self._sql_max_tokens = sql_max_tokens
        self._synth_max_tokens = synth_max_tokens
        self._last_sql: str | None = None
        self._warnings: list[str] = []

    def handle(
        self,
        *,
        question: str,
        channel_id: str,
        thread_ts: str,
        progress: Progress | None = None,
        directive: str | None = None,
    ) -> AgentResult:
        def report(stage: str) -> None:
            if progress is not None:
                progress(stage)

        self._warnings = []
        safe_question = redact(question)
        safe_directive = redact(directive) if directive else None
        history = self._conversation.history(channel_id=channel_id, thread_ts=thread_ts, limit=10)
        history_text = "\n".join(
            f"{message.role}: {redact(message.content)}" for message in history
        )

        if safe_directive:
            # Reviewer direction is authoritative: act on it, never re-clarify
            # (a "decline this request" must produce a decline, not a question).
            report("Following reviewer guidance")
            scope = {"needs_clarification": False, "questions": []}
        else:
            report("Scoping your question")
            scope = self._scope(
                safe_question,
                history_text,
                # Once a reviewer has directed the agent, stop clarifying: act on it.
                clarified=len(history) >= 8 or any(m.role == "system" for m in history),
            )
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
        prior_turns = [redact(message.content) for message in history if message.role == "user"]
        if prior_turns and prior_turns[-1].strip() == safe_question.strip():
            prior_turns.pop()  # the current question is already stored in the thread
        # Route on the thread's recent asks too, so sources named before a
        # clarification round-trip still match (e.g. "negative reviews").
        routing_input = "\n".join([*prior_turns[-3:], safe_question])
        matches = route_question(self._conn, self._embedder, routing_input, top_k=3)
        source_ids = [match["id"] for match in matches]
        evidence, sql = self._retrieve(safe_question, matches, history_text, report)

        report("Writing the answer")
        decision = self._synthesise(safe_question, evidence, directive=safe_directive)
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
            warnings=list(self._warnings),
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
            max_tokens=self._scope_max_tokens,
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
            primary = next(match for match in matches if match["id"] in SQL_SOURCES)
            tell(f"Checking {primary['name']}")
            block, listing_ids = self._sql_block(question, history_text)
            blocks.append(block)
            sql = self._last_sql

        if any(match["id"] == "guest_reviews" for match in matches):
            tell("Sampling guest reviews")
            sampled: list[dict] = []
            scope_note = ""
            if sql:
                # Cohort-wide sentiment/topic stats from the warehouse, so the
                # sample supplies colour and SQL supplies the aggregate numbers.
                try:
                    stats = cohort_review_stats(self._conn, cohort_sql=sql)
                except (psycopg.Error, ValueError) as error:
                    logger.warning("cohort review stats failed: %s", error)
                    stats = None
                if stats:
                    blocks.append(stats)
                # Full cohort: the display LIMIT is stripped inside
                # sample_reviews, so the draw spans every listing the cohort
                # query matches, not just the fetched page.
                try:
                    sampled = sample_reviews(
                        self._conn,
                        cohort_sql=sql,
                        sample_size=self._sample_size,
                        seed=self._sample_seed,
                    )
                except (psycopg.Error, ValueError) as error:
                    logger.warning("cohort review sampling failed: %s", error)
                    self._warnings.append("some guest reviews could not be sampled")
                if sampled:
                    scope_note = "drawn from the full cohort matched by the SQL query"
            if not sampled and listing_ids:
                # Fallback: the fetched page's ids are still an unbiased base
                # for the cohort that page represents.
                sampled = sample_reviews(
                    self._conn,
                    listing_ids=listing_ids,
                    sample_size=self._sample_size,
                    seed=self._sample_seed,
                )
                if sampled:
                    scope_note = f"from the {len(listing_ids)} scoped listings (fetched page)"
            if sampled:
                blocks.append(
                    f"Random sample of {len(sampled)} reviews {scope_note} (measurement base):\n"
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
            source = next(match for match in matches if match["id"] == "policy_kb")
            tell(f"Checking {source['name']}")
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
            source = next(match for match in matches if match["id"] == "data_dictionary")
            tell(f"Checking {source['name']}")
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
            candidate = generate_sql(
                self._llm,
                question,
                history_text,
                feedback=feedback,
                max_tokens=self._sql_max_tokens,
            )
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
        self._warnings.append("could not run a database query; answer may be incomplete")
        return "SQL attempt failed after one retry; a reviewer should provide the figure.", []

    def _synthesise(self, question: str, evidence: str, directive: str | None = None) -> dict:
        content = f"Question: {question}\n\nEvidence:\n{evidence}"
        if directive:
            content += f"\n\nReviewer direction (authoritative): {directive}"
        response = self._llm.complete(
            messages=[{"role": "user", "content": content}],
            model=model_for("synthesize"),
            system=SYNTH_SYSTEM,
            max_tokens=self._synth_max_tokens,
            task="synthesize",
        )
        decision = parse_decision(response.text, fallback=response.text)
        if response.truncated:
            logger.warning(
                "synthesize hit max_tokens=%d (%d output); answer may be cut off",
                self._synth_max_tokens,
                response.output_tokens,
            )
            decision["answer"] += "\n_(answer may be cut off)_"
        return decision
