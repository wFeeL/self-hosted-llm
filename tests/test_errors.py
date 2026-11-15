import pytest

from error_classifier.errors import (
    InvalidInputError,
    LlmBadResponseError,
    LlmTimeoutError,
    LlmUnavailableError,
    ModelOutputInvalidError,
    QueueFullError,
    RequestTooLargeError,
    ServiceError,
    error_body,
)


@pytest.mark.parametrize(
    ("cls", "code", "status"),
    [
        (InvalidInputError, "invalid_input", 400),
        (RequestTooLargeError, "request_too_large", 413),
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


def test_headers_ask_not_to_retry_after_a_finished_generation() -> None:
    assert ModelOutputInvalidError("x").headers() == {"x-should-retry": "false"}
    assert LlmTimeoutError("x").headers() == {"x-should-retry": "false"}
    assert ServiceError("x").headers() == {"x-should-retry": "false"}
    assert LlmUnavailableError("x").headers() == {}


def test_queue_full_keeps_retry_after_only() -> None:
    assert QueueFullError("busy", 7, param="p").headers() == {"Retry-After": "7"}


def test_param_is_optional() -> None:
    assert InvalidInputError("x").param is None
    assert InvalidInputError("x", param="messages").param == "messages"
    assert QueueFullError("busy", 7).param is None


@pytest.mark.parametrize(
    ("status", "kind"),
    [(400, "invalid_request_error"), (429, "invalid_request_error"), (500, "server_error")],
)
def test_error_body_type_follows_the_status(status: int, kind: str) -> None:
    assert error_body("c", "m", status, "p") == {
        "error": {"code": "c", "message": "m", "type": kind, "param": "p"}
    }
    assert error_body("c", "m", status)["error"] == {
        "code": "c",
        "message": "m",
        "type": kind,
        "param": None,
    }
