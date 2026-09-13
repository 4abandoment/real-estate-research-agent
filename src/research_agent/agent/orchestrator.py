"""The research agent: scope -> route -> retrieve -> synthesise -> escalate."""

import logging
from dataclasses import dataclass, field

import psycopg

from research_agent.agent.parsing import parse_decision, parse_scope
from research_agent.agent.prompts import SCOPE_SYSTEM, SYNTH_SYSTEM
from research_agent.agent.router import SQL_SOURCES, route_question
from research_agent.agent.sql import (
    SqlValidationError,
    format_rows,
    generate_sql,
    run_sql,
    validate_sql,
)
from research_agent.db import record_query
from research_agent.embeddings import Embedder
from research_agent.llm.client import LLMClient
from research_agent.llm.model_router import model_for
from research_agent.safety.pii import redact
from research_agent.search import search_policy, search_reviews

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResult:
    answer: str
    sql: str | None = None
    sources: list[str] = field(default_factory=list)
    needs_human: bool = False
    escalation_reason: str | None = None


class ResearchAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        conn,
        embedder: Embedder,
        conversation,
        sql_rows: int = 50,
    ) -> None:
        self._llm = llm
        self._conn = conn
        self._embedder = embedder
        self._conversation = conversation
        self._sql_rows = sql_rows
        self._last_sql: str | None = None

    def handle(self, *, question: str, channel_id: str, thread_ts: str) -> AgentResult:
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

        matches = route_question(self._conn, self._embedder, safe_question, top_k=3)
        source_ids = [match["id"] for match in matches]
        evidence, sql = self._retrieve(safe_question, matches, history_text)

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
        self, question: str, matches: list[dict], history_text: str
    ) -> tuple[str, str | None]:
        blocks: list[str] = []
        sql: str | None = None

        if any(match["id"] in SQL_SOURCES for match in matches):
            blocks.append(self._sql_block(question, history_text))
            sql = self._last_sql

        if any(match["id"] == "guest_reviews" for match in matches):
            hits = search_reviews(self._conn, self._embedder, question, top_k=5)
            if hits:
                blocks.append(
                    "Guest reviews:\n" + "\n".join(f"- {redact(hit['content'])}" for hit in hits)
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

        return "\n\n".join(blocks) or "No evidence retrieved.", sql

    def _sql_block(self, question: str, history_text: str) -> str:
        feedback = ""
        for attempt in range(2):
            candidate = generate_sql(self._llm, question, history_text, feedback=feedback)
            try:
                validated = validate_sql(candidate, max_rows=self._sql_rows)
                columns, rows = run_sql(self._conn, validated, max_rows=self._sql_rows)
            except (SqlValidationError, psycopg.Error) as error:
                logger.warning("SQL attempt %d failed: %s\nSQL: %s", attempt + 1, error, candidate)
                feedback = f"Previous attempt failed with: {error}\nPrevious SQL:\n{candidate}"
                continue
            self._last_sql = validated
            return f"SQL result:\n{format_rows(columns, rows)}"
        self._last_sql = None
        return "SQL attempt failed after one retry; a reviewer should provide the figure."

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
