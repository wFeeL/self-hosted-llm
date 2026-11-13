import pytest
from pydantic import ValidationError

from error_classifier.config import load_settings


def test_defaults_with_model_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    settings = load_settings()
    assert settings.llm_model == "m"
    assert settings.llm_base_url == "http://llm:8080/v1"
    assert settings.llm_probe_timeout_s == 2.0
    assert settings.log_level == "INFO"


def test_reads_all_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://h:9/v1")
    monkeypatch.setenv("LLM_MODEL", "x")
    monkeypatch.setenv("LLM_PROBE_TIMEOUT_S", "0.5")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = load_settings()
    assert settings.llm_base_url == "http://h:9/v1"
    assert settings.llm_model == "x"
    assert settings.llm_probe_timeout_s == 0.5
    assert settings.log_level == "DEBUG"


def test_empty_base_url_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "/")
    with pytest.raises(ValidationError, match="llm_base_url"):
        load_settings()


def test_trailing_slash_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "http://x:1/v1/")
    assert load_settings().llm_base_url == "http://x:1/v1"


def test_missing_model_fails() -> None:
    with pytest.raises(ValidationError, match="llm_model"):
        load_settings()


def test_empty_model_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "")
    with pytest.raises(ValidationError, match="llm_model"):
        load_settings()


@pytest.mark.parametrize("value", ["abc", "0", "-1"])
def test_bad_timeout_fails(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_PROBE_TIMEOUT_S", value)
    with pytest.raises(ValidationError, match="llm_probe_timeout_s"):
        load_settings()


def test_unknown_log_level_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValidationError, match="log_level"):
        load_settings()


def test_classification_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    settings = load_settings()
    assert settings.request_timeout_s == 120.0
    assert settings.max_input_chars == 3000
    assert settings.max_request_bytes == 262144
    assert settings.max_output_tokens == 256
    assert settings.llm_temperature == 0.0
    assert settings.llm_disable_thinking is False


def test_reads_classification_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "30")
    monkeypatch.setenv("MAX_INPUT_CHARS", "500")
    monkeypatch.setenv("MAX_REQUEST_BYTES", "7000")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "64")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("LLM_DISABLE_THINKING", "true")
    settings = load_settings()
    assert settings.request_timeout_s == 30.0
    assert settings.max_input_chars == 500
    assert settings.max_request_bytes == 7000
    assert settings.max_output_tokens == 64
    assert settings.llm_temperature == 0.7
    assert settings.llm_disable_thinking is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("REQUEST_TIMEOUT_S", "0"),
        ("MAX_INPUT_CHARS", "199"),
        ("MAX_INPUT_CHARS", "4001"),
        ("MAX_OUTPUT_TOKENS", "15"),
        ("MAX_OUTPUT_TOKENS", "2049"),
        ("LLM_TEMPERATURE", "-0.1"),
        ("LLM_TEMPERATURE", "2.1"),
    ],
)
def test_classification_bounds_fail(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=name.lower()):
        load_settings()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("REQUEST_TIMEOUT_S", "0.1"),
        ("MAX_INPUT_CHARS", "200"),
        ("MAX_INPUT_CHARS", "4000"),
        ("MAX_OUTPUT_TOKENS", "16"),
        ("MAX_OUTPUT_TOKENS", "2048"),
        ("LLM_TEMPERATURE", "0"),
        ("LLM_TEMPERATURE", "2"),
    ],
)
def test_classification_bounds_accept(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv(name, value)
    load_settings()


def test_request_bytes_must_cover_escaped_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("MAX_INPUT_CHARS", "3000")
    monkeypatch.setenv("MAX_REQUEST_BYTES", "36064")
    assert load_settings().max_request_bytes == 36064
    monkeypatch.setenv("MAX_REQUEST_BYTES", "36063")
    with pytest.raises(ValidationError, match="max_request_bytes"):
        load_settings()


def test_default_request_bytes_cover_the_largest_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("MAX_INPUT_CHARS", "4000")
    assert load_settings().max_request_bytes == 262144
