from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # No env_file: compose reads .env and passes variables to the container.
    model_config = SettingsConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    llm_base_url: str = "http://llm:8080/v1"
    llm_model: str = Field(min_length=1)
    llm_probe_timeout_s: float = Field(default=2.0, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_concurrency: int = Field(default=2, ge=1, le=64)
    max_queue: int = Field(default=4, ge=0, le=1000)
    queue_timeout_s: float = Field(default=30.0, gt=0, le=600)
    max_input_chars: int = Field(default=3000, ge=200, le=4000)
    max_request_bytes: int = 262144
    max_output_tokens: int = Field(default=256, ge=16, le=2048)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_disable_thinking: bool = False
    api_key: SecretStr | None = None

    @field_validator("api_key", mode="after")
    @classmethod
    def _check_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None or not value.get_secret_value():
            return None
        if not all(0x21 <= ord(char) <= 0x7E for char in value.get_secret_value()):
            raise ValueError("API_KEY must be printable ASCII without spaces")
        return value

    @field_validator("llm_base_url", mode="after")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        stripped = value.rstrip("/")
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @model_validator(mode="after")
    def _body_limit_covers_input(self) -> Self:
        # JSON escapes an emoji as two \uXXXX surrogates, 12 bytes for one character.
        if self.max_request_bytes < 12 * self.max_input_chars + 64:
            raise ValueError("max_request_bytes must be at least 12 * max_input_chars + 64")
        return self


def load_settings() -> Settings:
    return Settings.model_validate({})
