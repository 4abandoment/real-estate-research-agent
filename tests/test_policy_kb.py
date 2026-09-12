from research_agent.policy_kb import chunk


def test_chunk_respects_size_with_overlap() -> None:
    parts = chunk("x" * 1000, size=400, overlap=100)

    assert len(parts) >= 3
    assert all(len(part) <= 400 for part in parts)


def test_chunk_drops_tiny_fragments() -> None:
    assert chunk("short text", size=800, overlap=100) == []
