from research_agent.safety.pii import redact


def test_redacts_email() -> None:
    assert redact("contact me at a.b@example.com") == "contact me at [EMAIL]"


def test_redacts_uk_phone() -> None:
    assert redact("call 07700 900123") == "call [PHONE]"


def test_redacts_long_number() -> None:
    assert redact("account 123456789012") == "account [NUMBER]"


def test_keeps_plain_text() -> None:
    assert redact("the flat was lovely and bright") == "the flat was lovely and bright"


def test_handles_none() -> None:
    assert redact(None) == ""
