from research_agent.playbooks import embed_text, load_playbooks


def test_loads_all_playbooks() -> None:
    playbooks = load_playbooks()

    assert len(playbooks) == 6
    ids = {item["id"] for item in playbooks}
    assert {
        "leasing_warehouse",
        "transactions_ledger",
        "land_registry",
        "guest_reviews",
        "policy_kb",
        "data_dictionary",
    } <= ids


def test_playbooks_declare_retrieval_and_tables() -> None:
    for item in load_playbooks():
        assert item["retrieval"] in {"sql", "semantic", "metadata"}
        assert item.get("tables")


def test_embed_text_combines_name_and_description() -> None:
    assert embed_text({"name": "X", "description": "Y"}) == "X. Y"
