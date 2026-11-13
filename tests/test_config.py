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
