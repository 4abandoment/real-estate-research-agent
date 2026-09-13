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
from research_agent.llm.client import AnthropicClient

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
]

FORBIDDEN = [r"could not parse"]


def main() -> None:
    runs = next((int(a) for a in sys.argv[1:] if a.isdigit()), 2)
    settings = load_settings()
    if not settings.database_url or not settings.anthropic_api_key:
        raise SystemExit("DATABASE_URL and ANTHROPIC_API_KEY are required.")

    conn = connect(settings.database_url)
    apply_schema(conn)
    agent = ResearchAgent(
        llm=AnthropicClient(settings.anthropic_api_key),
        conn=conn,
        embedder=Embedder(),
        conversation=ConversationStore(conn),
        sample_seed=SEED,
    )

    lines = [f"# Eval report — {runs} run(s) per question, seed {SEED}", ""]
    lines.append("| question | check | pass |")
    lines.append("|---|---|---:|")
    total_checks = 0
    passed_checks = 0

    for name, question, patterns in QUESTIONS:
        answers = []
        for run in range(runs):
            thread = f"eval-{name}-{run}-{int(time.time())}"
            result = agent.handle(question=question, channel_id=EVAL_CHANNEL, thread_ts=thread)
            answers.append(result.answer)
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
