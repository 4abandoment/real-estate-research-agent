"""Summarise the token/cost usage log as markdown."""

from pathlib import Path

from research_agent.config import load_settings
from research_agent.llm.usage import usage_summary


def main() -> None:
    path = Path(load_settings().usage_log_path)
    print(usage_summary(path))


if __name__ == "__main__":
    main()
