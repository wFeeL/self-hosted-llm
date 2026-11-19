import re
from typing import Final

EMAIL_PATTERN: Final = re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[^\W\d_]{2,}")
EMAIL_MASK: Final = "<email>"


def mask_emails(text: str) -> str:
    return EMAIL_PATTERN.sub(EMAIL_MASK, text)
