"""Token and cost tracking for every LLM call."""

import functools
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from research_agent.config import load_settings

if TYPE_CHECKING:
    from research_agent.llm.client import LLMResponse

# ponytail: public list prices (USD per million tokens); update if rates change.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
}

T = TypeVar("T")


@dataclass(frozen=True)
class UsageEntry:
    ts: float
    task: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def log_usage(task: str, response: "LLMResponse") -> UsageEntry:
    entry = UsageEntry(
        ts=time.time(),
        task=task,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        cost_usd=estimate_cost(response.model, response.input_tokens, response.output_tokens),
    )
    path = Path(load_settings().usage_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry)) + "\n")
    return entry


def track_usage(task: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            response = func(*args, **kwargs)
            log_usage(task, response)
            return response

        return wrapper

    return decorator


def usage_totals(path: Path) -> tuple[int, float]:
    """Total calls and total estimated cost from a usage log."""
    if not path.exists():
        return 0, 0.0
    calls, cost = 0, 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        calls += 1
        cost += entry["cost_usd"]
    return calls, cost


def usage_summary(path: Path) -> str:
    """Markdown cost summary per model from a usage log."""
    if not path.exists():
        return f"No usage log at {path}"
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
