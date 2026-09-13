"""Map a task to a model.

Models are env-configurable so cheaper providers can be trialled per task
(``MODEL_SCOPE``, ``MODEL_SQL``, ``MODEL_SYNTH``) and eval-gated before they
become defaults. Ids prefixed with ``openrouter/`` route through OpenRouter;
anything else goes to Anthropic. Token budgets live in Settings
(``SCOPE_MAX_TOKENS`` etc.), not here.
"""

import os

CHEAP = "claude-haiku-4-5-20251001"
CAPABLE = "claude-sonnet-5"

_DEFAULT_MODELS = {
    "classify": CHEAP,
    "scope": CHEAP,
    "route": CHEAP,
    "sql": CAPABLE,
    "synthesize": CAPABLE,
}

_ENV_MODELS = {
    "classify": "MODEL_CLASSIFY",
    "scope": "MODEL_SCOPE",
    "route": "MODEL_ROUTE",
    "sql": "MODEL_SQL",
    "synthesize": "MODEL_SYNTH",
}


def model_for(task: str) -> str:
    default = _DEFAULT_MODELS.get(task, CAPABLE)
    override = os.getenv(_ENV_MODELS.get(task, ""), "").strip()
    return override or default
