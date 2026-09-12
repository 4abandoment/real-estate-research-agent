"""PII redaction for logs, prompts and committed samples.

Deliberately conservative: false positives are acceptable, leaking is not.
"""

import re

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+44\s?|0)(?:\d[\s-]?){9,11}(?!\d)")
LONG_DIGITS = re.compile(r"(?<!\d)\d{11,}(?!\d)")


def redact(text: str | None) -> str:
    if not text:
        return ""
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    return LONG_DIGITS.sub("[NUMBER]", text)
