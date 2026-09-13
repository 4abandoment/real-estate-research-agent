from research_agent.agent.router import required_sources


def test_review_mentions_force_guest_reviews() -> None:
    assert required_sources(
        "What do guests complain about most in listings in the top 20% of arrears?"
    ) == {"guest_reviews"}
    assert required_sources(
        "What kind of negative reviews are common in top decile properties?"
    ) == {"guest_reviews"}


def test_finance_questions_have_no_required_sources() -> None:
    assert required_sources("What is our overdue rent across the portfolio?") == set()
    assert required_sources("How do we protect a tenancy deposit?") == set()
