import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from bench_fakes import Stack
from conftest import Serve
from pydantic import ValidationError
from test_bench_cli import run_cli

from error_classifier.api.app import create_app
from error_classifier.config import Settings
from error_classifier.errors import UnauthorizedError

pytestmark = pytest.mark.anyio

KEY = "s3cret-Key_0123"
URL = "http://llm.test/v1/chat/completions"
ANSWER = {
    "category": "network_error",
    "severity": "high",
    "summary": "The upstream refused the connection.",
    "suggested_action": "Check that the upstream is running.",
}
CLASSIFY = ("/classify", {"message": "boom"})
CHAT = ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "boom"}]})
PROTECTED = [pytest.param(*CLASSIFY, id="classify"), pytest.param(*CHAT, id="chat")]
UNAUTHORIZED = {
    "error": {
        "code": "unauthorized",
        "message": "Missing or invalid API key",
        "type": "invalid_request_error",
        "param": None,
    }
}


def _settings(api_key: str | None = KEY) -> Settings:
    return Settings.model_validate(
        {"llm_model": "m", "llm_base_url": "http://llm.test/v1", "api_key": api_key}
    )


def _good() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "granite",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ANSWER)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def test_unauthorized_error_contract() -> None:
    error = UnauthorizedError()
    assert (error.code, error.status_code) == ("unauthorized", 401)
    assert error.message == "Missing or invalid API key"
    assert error.headers() == {"WWW-Authenticate": "Bearer"}


@pytest.mark.parametrize(("path", "body"), PROTECTED)
@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="missing"),
        pytest.param({"authorization": "Bearer wrong"}, id="wrong"),
        pytest.param({"authorization": f"Basic {KEY}"}, id="other-scheme"),
        pytest.param({"authorization": KEY}, id="no-scheme"),
        pytest.param({"authorization": b"Bearer \xe9"}, id="non-ascii-byte"),
    ],
)
async def test_bad_credentials_give_401(
    serve: Serve, path: str, body: dict[str, object], headers: dict[str, Any]
) -> None:
    async with serve(create_app(_settings())) as client:
        response = await client.post(path, json=body, headers=headers)
    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(("path", "body"), PROTECTED)
@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
async def test_right_key_passes(
    respx_mock: respx.MockRouter,
    serve: Serve,
    path: str,
    body: dict[str, object],
    scheme: str,
) -> None:
    respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings())) as client:
        response = await client.post(path, json=body, headers={"authorization": f"{scheme} {KEY}"})
    assert response.status_code == 200


async def test_models_need_the_key_too(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        denied = await client.get("/v1/models")
        allowed = await client.get("/v1/models", headers={"authorization": f"Bearer {KEY}"})
    assert (denied.status_code, allowed.status_code) == (401, 200)


async def test_health_and_ready_are_open(respx_mock: respx.MockRouter, serve: Serve) -> None:
    respx_mock.get("http://llm.test/v1/models").respond(200, json={"data": [{"id": "m"}]})
    async with serve(create_app(_settings())) as client:
        health = await client.get("/health")
        ready = await client.get("/ready")
    assert (health.status_code, ready.status_code) == (200, 200)


@pytest.mark.parametrize(("path", "body"), PROTECTED)
async def test_without_a_configured_key_nothing_changes(
    respx_mock: respx.MockRouter, serve: Serve, path: str, body: dict[str, object]
) -> None:
    respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings(None))) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200


@pytest.mark.parametrize("good", ["abc123", "a.b_c~d+e/f=g-h", "0" * 64])
def test_plain_keys_are_accepted(good: str) -> None:
    settings = Settings.model_validate({"llm_model": "m", "api_key": good})
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == good


def test_empty_key_means_no_key() -> None:
    assert Settings.model_validate({"llm_model": "m", "api_key": ""}).api_key is None


def test_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("API_KEY", KEY)
    settings = Settings.model_validate({})
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == KEY


@pytest.mark.parametrize(
    "bad",
    [
        "has space",
        "tab\tkey",
        "café",
        "line\nbreak",
        "pa$$word",
        'quo"te',
        "it's",
        "a#b",
        "back`tick",
    ],
)
def test_unprintable_key_is_rejected_without_echoing_it(bad: str) -> None:
    with pytest.raises(ValidationError) as caught:
        Settings.model_validate({"llm_model": "m", "api_key": bad})
    text = str(caught.value)
    assert "API_KEY may contain only letters, digits" in text
    assert bad not in text


def test_key_is_not_in_repr() -> None:
    settings = _settings()
    assert KEY not in repr(settings)
    assert KEY not in str(settings)


def test_warning_when_the_key_is_empty(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    create_app(_settings(None))
    messages = [r.getMessage() for r in caplog.records if r.name == "error_classifier.api"]
    assert messages == ["API_KEY is empty: the API accepts requests without a key"]


def test_no_warning_when_the_key_is_set(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    create_app(_settings())
    assert [r for r in caplog.records if r.name == "error_classifier.api"] == []


def test_benchmark_sends_the_key_to_classify_only(
    stack: Stack, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_KEY", KEY)
    code, _ = run_cli(stack, tmp_path / "out")
    assert code == 0
    assert stack.api_auth
    assert set(stack.api_auth) == {f"Bearer {KEY}"}
    assert stack.ready_auth
    assert set(stack.ready_auth) == {None}
    assert stack.llm_auth
    assert set(stack.llm_auth) == {None}


def test_benchmark_without_a_key_sends_no_header(stack: Stack, tmp_path: Path) -> None:
    code, _ = run_cli(stack, tmp_path / "out")
    assert code == 0
    assert set(stack.api_auth) == {None}


def test_benchmark_stops_on_401(
    stack: Stack,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stack.mode = lambda n: httpx.Response(401, json=UNAUTHORIZED)
    code, _ = run_cli(stack, tmp_path / "out")
    assert code == 3
    assert "the API answered 401: export API_KEY before the run" in capsys.readouterr().err
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())
