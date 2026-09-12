from research_agent.agent.parsing import parse_decision, parse_scope


def test_parse_scope_detects_clarification() -> None:
    result = parse_scope('{"needs_clarification": true, "questions": ["Which area?"]}')
    assert result == {"needs_clarification": True, "questions": ["Which area?"]}


def test_parse_scope_defaults_on_garbage() -> None:
    assert parse_scope("no json here") == {"needs_clarification": False, "questions": []}


def test_parse_scope_ignores_empty_questions() -> None:
    result = parse_scope('{"needs_clarification": true, "questions": []}')
    assert result["needs_clarification"] is False


def test_parse_decision_reads_delimited_fields() -> None:
    text = (
        "ANSWER: Overdue rent is £100.\nSpanning two lines.\n"
        "CONFIDENCE: 0.9\nNEEDS_HUMAN: false\nREASON:"
    )
    result = parse_decision(text, fallback="x")
    assert "Overdue rent is £100" in result["answer"]
    assert "Spanning two lines." in result["answer"]
    assert result["needs_human"] is False
    assert result["confidence"] == 0.9


def test_parse_decision_escalates_when_flagged() -> None:
    text = (
        "ANSWER: Not enough evidence.\nCONFIDENCE: 0.3\nNEEDS_HUMAN: true\nREASON: low confidence"
    )
    result = parse_decision(text, fallback="x")
    assert result["needs_human"] is True
    assert result["reason"] == "low confidence"


def test_parse_decision_falls_back_to_json() -> None:
    data = '{"answer": "42", "confidence": 0.9, "needs_human": false, "reason": ""}'
    result = parse_decision(data, fallback="x")
    assert result["answer"] == "42"
    assert result["needs_human"] is False


def test_parse_decision_keeps_plain_text_without_escalating() -> None:
    result = parse_decision("plain text answer", fallback="plain text answer")
    assert result["needs_human"] is False
    assert result["answer"] == "plain text answer"


def test_parse_decision_escalates_on_empty_response() -> None:
    result = parse_decision("", fallback="nothing")
    assert result["needs_human"] is True
    assert result["reason"] == "empty model response"
