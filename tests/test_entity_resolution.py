from research_agent.entity_resolution import best_match, normalise


def test_normalise_strips_borough_prefixes() -> None:
    assert normalise("London Borough of Camden") == "camden"
    assert normalise("Royal Borough of Kensington and Chelsea") == "kensington and chelsea"


def test_normalise_removes_punctuation() -> None:
    assert normalise("Barking & Dagenham") == "barking dagenham"


def test_best_match_prefers_exact() -> None:
    match, score = best_match("camden", ["Camden", "Hackney"])
    assert match == "Camden"
    assert score == 1.0


def test_best_match_fuzzy() -> None:
    match, score = best_match("kensington and chelsea", ["Kensington & Chelsea", "Camden"])
    assert match == "Kensington & Chelsea"
    assert score >= 0.86


def test_best_match_returns_none_below_threshold() -> None:
    match, _ = best_match("camden", ["Hackney", "Islington"])
    assert match is None
