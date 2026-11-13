import pytest

from error_classifier.errors import (
    InvalidInputError,
    LlmBadResponseError,
    LlmTimeoutError,
    ModelOutputInvalidError,
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
