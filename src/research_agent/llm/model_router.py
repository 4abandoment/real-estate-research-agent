"""Map a task to the cheapest capable model."""

CHEAP = "claude-haiku-4-5-20251001"
CAPABLE = "claude-sonnet-5"

_TASK_MODELS = {
    "classify": CHEAP,
    "scope": CHEAP,
    "route": CHEAP,
    "sql": CAPABLE,
    "synthesize": CAPABLE,
}


def model_for(task: str) -> str:
    return _TASK_MODELS.get(task, CAPABLE)
