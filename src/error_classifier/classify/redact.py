import re
from typing import Final

EMAIL_PATTERN: Final = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)
EMAIL_MASK: Final = "<email>"


def mask_emails(text: str) -> str:
    return EMAIL_PATTERN.sub(EMAIL_MASK, text)
