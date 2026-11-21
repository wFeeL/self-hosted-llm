from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SelectSettings(BaseSettings):
    # No env_file: compose reads .env and passes variables to the container.
    model_config = SettingsConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    select_base_url: str = "http://llm-select:8080"
    main_llm_url: str = "http://llm:8080"
    presets_path: Path = Path("/app/select/presets.ini")
    data_root: Path = Path("/app")  # eval/ and bench/results/ live here
    select_load_timeout_s: float = Field(default=1800.0, gt=0)
    select_target_p95_s: float = Field(default=30.0, gt=0)
    select_model_max_s: float = Field(default=1200.0, gt=0)
    request_timeout_s: float = Field(default=120.0, gt=0)
    queue_timeout_s: float = Field(default=30.0, gt=0)
    max_input_chars: int = Field(default=3000, ge=200, le=4000)
    max_output_tokens: int = Field(default=256, ge=16, le=2048)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_mem_limit: str = "6656m"
    api_mem_limit: str = "512m"
    llm_cpus: str = "6"
    llama_arg_threads: int = 6
    llama_arg_ctx_size: int = 8192
    llama_arg_n_parallel: int = 2
