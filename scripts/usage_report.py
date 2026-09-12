"""Summarise the token/cost usage log as markdown."""

import json
from collections import defaultdict
from pathlib import Path

from research_agent.config import load_settings


def summarise(path: Path) -> str:
    totals: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0, 0])
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        row = totals[entry["model"]]
        row[0] += entry["input_tokens"]
        row[1] += entry["output_tokens"]
        row[2] += entry["cost_usd"]
        row[3] += 1

    lines = ["| model | calls | input | output | cost (USD) |", "|---|---:|---:|---:|---:|"]
    for model, (input_tokens, output_tokens, cost, calls) in sorted(totals.items()):
        lines.append(
            f"| {model} | {calls} | {int(input_tokens)} | {int(output_tokens)} | {cost:.4f} |"
        )
    total_cost = sum(row[2] for row in totals.values())
    lines.append(f"\n**Total: ${total_cost:.4f}**")
    return "\n".join(lines)


def main() -> None:
    path = Path(load_settings().usage_log_path)
    if not path.exists():
        print(f"No usage log at {path}")
        return
    print(summarise(path))


if __name__ == "__main__":
    main()
