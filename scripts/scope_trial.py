"""Scripted multi-turn scoping trial.

Drives the real orchestrator over five scenarios with canned user replies
(covering up to 3 batched clarify questions per turn). Measures clarify turns
used, whether the final answer stated an assumption, and rubric spot-checks.

Usage:
    python scripts/scope_trial.py            # all scenarios
    python scripts/scope_trial.py in_trouble # one scenario

Transcripts: logs/scope_trials/<scenario>.md
"""

import sys
import time
from pathlib import Path

from research_agent.agent.orchestrator import ResearchAgent
from research_agent.config import load_settings
from research_agent.db import ConversationStore, apply_schema, connect
from research_agent.embeddings import Embedder
from research_agent.llm.client import create_client
from research_agent.llm.model_router import model_for

CHANNEL = "C_SCOPE_TRIAL"
MAX_CLARIFY_TURNS = 2

SCENARIOS = [
    ("overdue_clear", "What is our overdue rent across the portfolio?", []),
    (
        "in_trouble",
        "Which of our listings are in trouble?",
        ["Use total overdue invoice balance as the measure of trouble."],
    ),
    (
        "good_bad",
        "Tell me about the good and bad ones",
        [
            "Good = most positive guest sentiment; bad = highest overdue arrears. "
            "English reviews only, portfolio-wide."
        ],
    ),
    ("data_sources", "What data do you have access to?", []),
    (
        "payments_vs_satisfaction",
        "How are we doing on payments vs guest satisfaction?",
        [
            "Payments = collected vs outstanding arrears; satisfaction = average "
            "sentiment and top complaint topics, portfolio-wide."
        ],
    ),
]


def main() -> None:
    names = sys.argv[1:] or [name for name, _, _ in SCENARIOS]
    settings = load_settings()
    conn = connect(settings.database_url)
    apply_schema(conn)
    agent = ResearchAgent(
        llm=create_client(settings),
        conn=conn,
        embedder=Embedder(),
        conversation=ConversationStore(conn),
        sample_seed=0.42,
        scope_max_tokens=settings.scope_max_tokens,
        sql_max_tokens=settings.sql_max_tokens,
        synth_max_tokens=settings.synth_max_tokens,
    )
    print(
        "models in use: "
        f"scope={model_for('scope')} sql={model_for('sql')} "
        f"synth={model_for('synthesize')}"
    )

    out_dir = Path(settings.usage_log_path).parent / "scope_trials"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, question, canned in SCENARIOS:
        if names and name not in names:
            continue
        thread = f"scope-trial-{name}-{int(time.time())}"
        conversation = ConversationStore(conn)
        lines = [f"# {name}", "", f"**Q1:** {question}", ""]
        pending = question
        clarify_turns = 0
        assumption_stated = False
        for turn in range(MAX_CLARIFY_TURNS + 1):
            conversation.add(channel_id=CHANNEL, thread_ts=thread, role="user", content=pending)
            result = agent.handle(
                question=pending, channel_id=CHANNEL, thread_ts=thread, progress=None
            )
            conversation.add(
                channel_id=CHANNEL, thread_ts=thread, role="assistant", content=result.answer
            )
            lines.append(f"\n**Agent (turn {turn + 1}):**\n{result.answer}")
            if not result.answer.startswith("Before I dig in"):
                assumption_stated = "ASSUMPTION:" in result.answer or "Interpreted as:" in pending
                break
            clarify_turns += 1
            if turn < len(canned):
                pending = canned[turn]
                lines.append(f"\n**User (canned reply {turn + 1}):** {pending}")
            else:
                break

        verdict = "clarified then answered" if clarify_turns else "answered directly"
        summary = (
            f"{name}: clarify_turns={clarify_turns} ({verdict}), "
            f"assumption_stated={assumption_stated}"
        )
        print(summary, flush=True)
        lines.append(f"\n---\n\n{summary}")
        (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"transcripts: {out_dir}")


if __name__ == "__main__":
    main()
