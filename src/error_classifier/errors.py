from typing import ClassVar


class ServiceError(Exception):
    code: ClassVar[str] = "internal_error"
    status_code: ClassVar[int] = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LlmUnavailableError(ServiceError):
    code = "llm_unavailable"
    status_code = 503
