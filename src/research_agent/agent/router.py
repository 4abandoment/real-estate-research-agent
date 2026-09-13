"""Source prioritisation via the playbook registry."""

from research_agent.embeddings import Embedder
from research_agent.playbooks import match_sources

SQL_SOURCES = {"leasing_warehouse", "transactions_ledger", "land_registry"}

REVIEW_HINTS = ("review", "guest", "complain", "feedback", "sentiment")


def required_sources(question: str) -> set[str]:
    """Sources explicitly named in the question are a floor, not a suggestion.

    Long finance-heavy questions dilute the review signal below top-k, so an
    explicit mention must always route regardless of embedding rank.
    """
    text = question.lower()
    if any(hint in text for hint in REVIEW_HINTS):
        return {"guest_reviews"}
    return set()


def route_question(conn, embedder: Embedder, question: str, top_k: int = 3) -> list[dict]:
    return match_sources(
        conn, embedder, question, top_k=top_k, required_ids=required_sources(question)
    )
