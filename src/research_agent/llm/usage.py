"""Token and cost tracking for every LLM call."""

import functools
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from research_agent.config import load_settings
from research_agent.llm.client import LLMResponse

# ponytail: public list prices (USD per million tokens); update if rates change.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
}

T = TypeVar("T", bound=LLMResponse)


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


def log_usage(task: str, response: LLMResponse) -> UsageEntry:
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
