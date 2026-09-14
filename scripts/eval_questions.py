"""Repeatable evaluation of agent answer quality.

Fixed question set x N runs, seeded random sampling, and a mechanical rubric
(basis stated, cutoff cited, quotes carry ids, sampling method named, no
parse failures). Prints a pass-rate table and writes logs/eval_report.md.

Usage:
    python scripts/eval_questions.py [runs]     # default 2 runs per question

Process rule (AGENTS.md): run this before merging any prompt or agent-pipeline
change, and paste the report into the PR body.
"""

import re
import sys
import time
from pathlib import Path

from research_agent.agent.orchestrator import ResearchAgent
from research_agent.config import load_settings
from research_agent.db import ConversationStore, apply_schema, connect
from research_agent.embeddings import Embedder
from research_agent.llm.client import create_client

SEED = 0.42
EVAL_CHANNEL = "C_EVAL"

# (name, question, [required regex patterns, case-insensitive])
QUESTIONS = [
    (
        "top_decile",
        "What are the most common issues reported by guests in the top 10%"
        " most expensive airbnb locations?",
        [r"basis", r"random sample", r"caveat", r"£|\bGBP\b|\bprice"],
    ),
    (
        "overdue",
        "What is our overdue rent across the portfolio?",
        [r"basis", r"£"],
    ),
    (
        "soho_negative",
        "Tell me about negative reviews in Soho",
        [r"basis", r"random sample|illustrative", r"#\d+|listing \d+"],
    ),
    (
        "deposit",
        "How long do we have to protect a tenancy deposit?",
        [r"30\s*-?\s*day", r"gov\.uk"],
    ),
    (
        "data_dictionary",
        "What data do you have access to?",
        [r"listings", r"calendar"],
    ),
    (
        "arrears_reviews",
        "What do guests complain about most in listings with the 20% highest"
        " total overdue invoice balances?",
        [r"basis", r"random sample", r"#\d+|listing \d+"],
    ),
    (
        "in_trouble",
        "Which of our listings are in trouble?",
        # Ambiguity must be surfaced: either a clarifying question or a stated
        # assumption - never a silent interpretation.
        [r"before i dig in|assum"],
    ),
]

# Questions that must NOT clarify (scope sensitivity guard: no over-clarifying
# of well-posed questions).
SCOPE_FORBIDDEN = {
    "overdue": [r"^before i dig in"],
    "deposit": [r"^before i dig in"],
    "data_dictionary": [r"^before i dig in"],
}

# Scripted user replies for questions whose first turn is a clarification;
# the rubric runs against the final answer after the canned replies.
QUESTION_SCRIPTS = {
    "top_decile": [
        "Most expensive by nightly listing price; common issues = the most"
        " frequently flagged review topics."
    ],
    "soho_negative": [
        "All listings in Soho; show a sample of negative reviews with their review ids."
    ],
}

FORBIDDEN = [r"could not parse"]


def main() -> None:
    runs = next((int(a) for a in sys.argv[1:] if a.isdigit()), 2)
    settings = load_settings()
    if not settings.database_url or not settings.anthropic_api_key:
        raise SystemExit("DATABASE_URL and ANTHROPIC_API_KEY are required.")

    from research_agent.llm.model_router import model_for

    print(
        "models in use: "
        f"scope={model_for('scope')} sql={model_for('sql')} synth={model_for('synthesize')}"
    )

    conn = connect(settings.database_url)
    apply_schema(conn)
    conversation = ConversationStore(conn)
    agent = ResearchAgent(
        llm=create_client(settings),
        conn=conn,
        embedder=Embedder(),
        conversation=conversation,
        sample_seed=SEED,
        scope_max_tokens=settings.scope_max_tokens,
        sql_max_tokens=settings.sql_max_tokens,
        synth_max_tokens=settings.synth_max_tokens,
    )

    lines = [f"# Eval report — {runs} run(s) per question, seed {SEED}", ""]
    lines.append("| question | check | pass |")
    lines.append("|---|---|---:|")
    total_checks = 0
    passed_checks = 0

    answers_dir = Path(settings.usage_log_path).parent / "eval_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    for name, question, patterns in QUESTIONS:
        scripted = QUESTION_SCRIPTS.get(name, [])
        answers = []
        for run in range(runs):
            thread = f"eval-{name}-{run}-{int(time.time())}"
            pending = question
            final_answer = None
            transcript_lines = [f"# {name} (run {run})", "", f"**Q1:** {question}", ""]
            for turn in range(len(scripted) + 1):
                conversation.add(
                    channel_id=EVAL_CHANNEL, thread_ts=thread, role="user", content=pending
                )
                result = agent.handle(question=pending, channel_id=EVAL_CHANNEL, thread_ts=thread)
                conversation.add(
                    channel_id=EVAL_CHANNEL,
                    thread_ts=thread,
                    role="assistant",
                    content=result.answer,
                )
                transcript_lines.append(f"\n**Agent (turn {turn + 1}):**\n{result.answer}")
                final_answer = result.answer
                if not result.answer.startswith("Before I dig in"):
                    break
                if turn < len(scripted):
                    pending = scripted[turn]
                    transcript_lines.append(f"\n**User (scripted reply {turn + 1}):** {pending}")
            answers.append(final_answer)
            (answers_dir / f"{name}_run{run}.md").write_text(
                "\n".join(transcript_lines) + "\n", encoding="utf-8"
            )
        for pattern in patterns:
            hits = sum(1 for answer in answers if re.search(pattern, answer, re.IGNORECASE))
            total_checks += runs
            passed_checks += hits
            lines.append(f"| {name} | /{pattern}/ | {hits}/{runs} |")
        for pattern in FORBIDDEN:
            hits = sum(1 for answer in answers if re.search(pattern, answer, re.IGNORECASE))
            total_checks += runs
            passed_checks += runs - hits
            lines.append(f"| {name} | !/{pattern}/ absent | {runs - hits}/{runs} |")
        for pattern in SCOPE_FORBIDDEN.get(name, []):
            hits = sum(
                1
                for answer in answers
                if re.search(pattern, answer.strip(), re.IGNORECASE | re.MULTILINE)
            )
            total_checks += runs
            passed_checks += runs - hits
            lines.append(f"| {name} | !/{pattern}/ absent (scope) | {runs - hits}/{runs} |")
        print(f"{name}: done ({runs} runs)")

    rate = 100 * passed_checks / total_checks if total_checks else 0.0
    lines.append("")
    lines.append(f"**Overall: {passed_checks}/{total_checks} ({rate:.0f}%)**")
    report = "\n".join(lines)
    print(f"\n{report}\n")

    out = Path(settings.usage_log_path).parent / "eval_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
