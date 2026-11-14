from typing import ClassVar


class ServiceError(Exception):
    code: ClassVar[str] = "internal_error"
    status_code: ClassVar[int] = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def headers(self) -> dict[str, str]:
        return {}


class LlmUnavailableError(ServiceError):
    code = "llm_unavailable"
    status_code = 503


class InvalidInputError(ServiceError):
    code = "invalid_input"
    status_code = 400


class ModelOutputInvalidError(ServiceError):
    code = "model_output_invalid"
    status_code = 502


class LlmBadResponseError(ServiceError):
    code = "llm_bad_response"
    status_code = 502


class LlmTimeoutError(ServiceError):
    code = "llm_timeout"
    status_code = 504


class QueueFullError(ServiceError):
    code = "queue_full"
    status_code = 429

    def __init__(self, message: str, retry_after_s: int) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s

    def headers(self) -> dict[str, str]:
        return {"Retry-After": str(self.retry_after_s)}
