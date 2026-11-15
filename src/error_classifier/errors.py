from typing import ClassVar


class ServiceError(Exception):
    code: ClassVar[str] = "internal_error"
    status_code: ClassVar[int] = 500

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param

    def headers(self) -> dict[str, str]:
        # The service already retried generation, or the wait was a full timeout.
        if self.status_code in (500, 502, 504):
            return {"x-should-retry": "false"}
        return {}


class LlmUnavailableError(ServiceError):
    code = "llm_unavailable"
    status_code = 503


class InvalidInputError(ServiceError):
    code = "invalid_input"
    status_code = 400


class ModelNotFoundError(ServiceError):
    code = "model_not_found"
    status_code = 404


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

    def __init__(self, message: str, retry_after_s: int, param: str | None = None) -> None:
        super().__init__(message, param)
        self.retry_after_s = retry_after_s

    def headers(self) -> dict[str, str]:
        return {**super().headers(), "Retry-After": str(self.retry_after_s)}


def error_body(
    code: str, message: str, status_code: int, param: str | None = None
) -> dict[str, object]:
    kind = "invalid_request_error" if status_code < 500 else "server_error"
    return {"error": {"code": code, "message": message, "type": kind, "param": param}}
