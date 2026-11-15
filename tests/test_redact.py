import pytest

from error_classifier.classify.redact import mask_emails

CASES = [
    ("rejected maria.ivanova@example.com because", "rejected <email> because"),
    ("write to ops@corp.example.org.", "write to <email>."),
    ("<a.b@x.io>", "<<email>>"),
    ("mailto:x@y.org", "mailto:<email>"),
    ("check the key for git@github.com", "check the key for <email>"),
    ("user@localhost failed", "user@localhost failed"),
    ("github.com/jackc/pgx/v5@v5.5.0", "github.com/jackc/pgx/v5@v5.5.0"),
    ("lodash@4.17.21", "lodash@4.17.21"),
    ("two a@b.cc and c+d@e.co.uk", "two <email> and <email>"),
    ("no address here", "no address here"),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_mask_emails(text: str, expected: str) -> None:
    assert mask_emails(text) == expected


@pytest.mark.parametrize("text", [case[0] for case in CASES])
def test_mask_is_idempotent(text: str) -> None:
    assert mask_emails(mask_emails(text)) == mask_emails(text)
