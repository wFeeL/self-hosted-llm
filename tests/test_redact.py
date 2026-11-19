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
    # Non-ASCII cases use \u escapes: ruff flags Cyrillic lookalikes in literals.
    (
        (
            "\u043f\u0438\u0441\u044c\u043c\u043e "
            "\u0438\u0432\u0430\u043d@\u043f\u043e\u0447\u0442\u0430.\u0440\u0444 "
            "\u043f\u0440\u0438\u0448\u043b\u043e"
        ),
        "\u043f\u0438\u0441\u044c\u043c\u043e <email> \u043f\u0440\u0438\u0448\u043b\u043e",
    ),
    ("ivan@\u043f\u043e\u0447\u0442\u0430.\u0440\u0444", "<email>"),
    ("\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c@mail.ru", "<email>"),
    ("\u0444\u0430\u0439\u043b@1.2.3", "\u0444\u0430\u0439\u043b@1.2.3"),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_mask_emails(text: str, expected: str) -> None:
    assert mask_emails(text) == expected


@pytest.mark.parametrize("text", [case[0] for case in CASES])
def test_mask_is_idempotent(text: str) -> None:
    assert mask_emails(mask_emails(text)) == mask_emails(text)
