from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # No env_file: compose reads .env and passes variables to the container.
    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    llm_base_url: str = "http://llm:8080/v1"
    llm_model: str = Field(min_length=1)
    llm_probe_timeout_s: float = Field(default=2.0, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("llm_base_url", mode="after")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        stripped = value.rstrip("/")
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


def load_settings() -> Settings:
    return Settings.model_validate({})
