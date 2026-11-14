import pytest

from error_classifier.errors import (
    InvalidInputError,
    LlmBadResponseError,
    LlmTimeoutError,
    ModelOutputInvalidError,
    QueueFullError,
    ServiceError,
)


@pytest.mark.parametrize(
    ("cls", "code", "status"),
    [
        (InvalidInputError, "invalid_input", 400),
        (ModelOutputInvalidError, "model_output_invalid", 502),
        (LlmBadResponseError, "llm_bad_response", 502),
        (LlmTimeoutError, "llm_timeout", 504),
    ],
)
def test_error_class_contract(cls: type[ServiceError], code: str, status: int) -> None:
    error = cls("text")
    assert error.code == code
    assert error.status_code == status
    assert error.message == "text"


def test_queue_full_error_contract() -> None:
    error = QueueFullError("busy", 7)
    assert error.code == "queue_full"
    assert error.status_code == 429
    assert error.message == "busy"
    assert error.retry_after_s == 7
    assert error.headers() == {"Retry-After": "7"}


def test_other_errors_have_no_headers() -> None:
    assert InvalidInputError("x").headers() == {}
