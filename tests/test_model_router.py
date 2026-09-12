from research_agent.llm.model_router import CAPABLE, CHEAP, model_for


def test_cheap_tasks_use_cheap_model() -> None:
    assert model_for("scope") == CHEAP
    assert model_for("route") == CHEAP


def test_capable_tasks_use_capable_model() -> None:
    assert model_for("sql") == CAPABLE
    assert model_for("synthesize") == CAPABLE


def test_unknown_task_defaults_to_capable() -> None:
    assert model_for("mystery") == CAPABLE
