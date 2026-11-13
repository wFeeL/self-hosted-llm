import json
import logging
import re
from collections.abc import AsyncGenerator

import httpx
import pytest
import respx
from conftest import Serve

from error_classifier.api.app import create_app
from error_classifier.classify.prompt import INPUT_PREFIX
from error_classifier.classify.schema import Category, ClassificationResult, Severity
from error_classifier.classify.service import Classification, Classifier, ClassifyStats
from error_classifier.config import Settings

pytestmark = pytest.mark.anyio

URL = "http://llm.test/v1/chat/completions"
LOGGER = "error_classifier.classify"
ANSWER: dict[str, str] = {
    "category": "network_error",
    "severity": "high",
    "summary": "The upstream refused the connection.",
    "suggested_action": "Check that the upstream is running.",
}
EXPECTED_200: dict[str, object] = {
    **ANSWER,
    "model": "granite",
    "truncated": False,
}


def _settings(**changes: object) -> Settings:
    values: dict[str, object] = {"llm_model": "m", "llm_base_url": "http://llm.test/v1"}
    return Settings.model_validate({**values, **changes})


def _runtime(content: str, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "granite",
            "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def _good() -> httpx.Response:
    return _runtime(json.dumps(ANSWER))


async def _post(client: httpx.AsyncClient, message: object) -> httpx.Response:
    return await client.post("/classify", json={"message": message})


def _info_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.INFO]


def _error_body(code: str, message: str) -> dict[str, object]:
    return {"error": {"code": code, "message": message}}


async def test_classify_200(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings())) as client:
        response = await _post(client, "ECONNREFUSED 10.0.0.5:5432")
    assert response.status_code == 200
    body = response.json()
    latency = body.pop("latency_ms")
    assert isinstance(latency, int)
    assert latency >= 0
    assert body == EXPECTED_200
    assert route.call_count == 1


async def test_classify_retry_then_200(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.post(URL).mock(side_effect=[_runtime("not json"), _good()])
    async with serve(create_app(_settings())) as client:
        response = await _post(client, "boom")
    assert response.status_code == 200
    assert response.json()["category"] == "network_error"
    assert route.call_count == 2


async def test_classify_502_after_two_invalid(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.post(URL).mock(side_effect=[_runtime("a"), _runtime("b")])
    async with serve(create_app(_settings())) as client:
        response = await _post(client, "boom")
    assert response.status_code == 502
    assert response.json() == _error_body(
        "model_output_invalid", "The model returned an invalid answer twice"
    )
    assert route.call_count == 2


@pytest.mark.parametrize(
    ("outcome", "status", "code"),
    [
        pytest.param(httpx.ReadTimeout("slow"), 504, "llm_timeout", id="timeout"),
        pytest.param(httpx.ConnectError("refused"), 503, "llm_unavailable", id="connect"),
        pytest.param(httpx.Response(503), 503, "llm_unavailable", id="503"),
        pytest.param(
            httpx.Response(400, json={"error": {"type": "exceed_context_size_error"}}),
            400,
            "invalid_input",
            id="context-overflow",
        ),
        pytest.param(
            httpx.Response(400, json={"error": {"type": "invalid_request_error"}}),
            502,
            "llm_bad_response",
            id="400-other",
        ),
    ],
)
async def test_runtime_failures(
    respx_mock: respx.MockRouter,
    serve: Serve,
    outcome: httpx.Response | Exception,
    status: int,
    code: str,
) -> None:
    route = respx_mock.post(URL).mock(side_effect=[outcome])
    async with serve(create_app(_settings())) as client:
        response = await _post(client, "boom")
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert route.call_count == 1


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize(
    ("content", "message"),
    [
        pytest.param(b'{"message": ""}', "Field 'message' must not be empty", id="empty"),
        pytest.param(b'{"message": "  \\n "}', "Field 'message' must not be empty", id="blank"),
        pytest.param(b"not json", "Invalid request: body: json_invalid", id="not-json"),
        pytest.param(b"{}", "Invalid request: message: missing", id="no-message"),
        pytest.param(b'{"message": 5}', "Invalid request: message: string_type", id="not-string"),
    ],
)
async def test_invalid_input(
    respx_mock: respx.MockRouter, serve: Serve, content: bytes, message: str
) -> None:
    route = respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings())) as client:
        response = await client.post("/classify", content=content)
    assert response.status_code == 400
    assert response.json() == _error_body("invalid_input", message)
    assert route.call_count == 0


def _small_body_settings() -> Settings:
    return _settings(max_input_chars=200, max_request_bytes=2500)


@pytest.mark.respx(assert_all_called=False)
async def test_body_too_large_with_content_length(
    respx_mock: respx.MockRouter, serve: Serve
) -> None:
    route = respx_mock.post(URL).mock(return_value=_good())
    body = b'{"message": "' + b"a" * 3000 + b'"}'
    async with serve(create_app(_small_body_settings())) as client:
        response = await client.post("/classify", content=body)
    assert response.status_code == 400
    assert response.json() == _error_body("invalid_input", "Request body is larger than 2500 bytes")
    assert route.call_count == 0


@pytest.mark.respx(assert_all_called=False)
async def test_body_too_large_streamed(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.post(URL).mock(return_value=_good())

    async def chunks() -> AsyncGenerator[bytes]:
        yield b'{"message": "'
        for _ in range(30):
            yield b"a" * 100
        yield b'"}'

    stream = chunks()
    try:
        async with serve(create_app(_small_body_settings())) as client:
            response = await client.post("/classify", content=stream)
    finally:
        await stream.aclose()
    assert "content-length" not in response.request.headers
    assert response.status_code == 400
    assert response.json() == _error_body("invalid_input", "Request body is larger than 2500 bytes")
    assert route.call_count == 0


async def test_long_message_truncated(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings())) as client:
        response = await _post(client, "a" * 5000)
    assert response.status_code == 200
    assert response.json()["truncated"] is True
    sent = json.loads(route.calls.last.request.content)
    last = sent["messages"][-1]["content"]
    assert "characters omitted" in last
    assert last.startswith(INPUT_PREFIX)
    assert len(json.loads(last.removeprefix(INPUT_PREFIX))) <= 3000


async def test_settings_reach_runtime(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.post(URL).mock(return_value=_good())
    settings = _settings(llm_temperature=0.5, max_output_tokens=100, llm_disable_thinking=True)
    async with serve(create_app(settings)) as client:
        await _post(client, "boom")
    sent = json.loads(route.calls.last.request.content)
    assert sent["temperature"] == 0.5
    assert sent["max_tokens"] == 100
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["model"] == "m"
    assert route.calls.last.request.extensions["timeout"]["read"] == 120.0


def _boom_classify(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(self: Classifier, message: str, stats: ClassifyStats) -> None:
        raise RuntimeError("secret-marker")

    monkeypatch.setattr(Classifier, "classify", boom)


async def test_unexpected_exception_500(
    respx_mock: respx.MockRouter,
    serve: Serve,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="error_classifier")
    _boom_classify(monkeypatch)
    async with serve(create_app(_settings())) as client:
        response = await _post(client, "boom")
    assert response.status_code == 500
    assert response.json() == _error_body("internal_error", "Internal error")
    assert "secret-marker" not in response.text
    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert all("secret-marker" not in r.getMessage() for r in loud)
    assert all(r.exc_info is None for r in loud)
    info = _info_records(caplog)
    assert len(info) == 1
    assert "outcome=internal_error" in info[0].getMessage()
    assert any(r.levelno == logging.ERROR and "RuntimeError" in r.getMessage() for r in loud)


async def test_log_has_no_user_text(
    respx_mock: respx.MockRouter, serve: Serve, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="error_classifier")
    echo = json.dumps({"category": "bogus", "summary": "SECRET-7f3a", "x": "SECRET-7f3a"})
    respx_mock.post(URL).mock(side_effect=[_runtime(echo), _good()])
    async with serve(create_app(_settings())) as client:
        response = await _post(client, "failure SECRET-7f3a at line 3")
    assert response.status_code == 200
    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert all("SECRET-7f3a" not in r.getMessage() for r in loud)
    info = _info_records(caplog)
    assert len(info) == 1
    text = info[0].getMessage()
    assert "outcome=ok" in text
    assert "attempts=2" in text


async def test_log_line_fields_on_success(
    respx_mock: respx.MockRouter, serve: Serve, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings())) as client:
        await _post(client, "  boom ")
    (record,) = _info_records(caplog)
    assert re.fullmatch(
        r"classify outcome=ok latency_ms=\d+ model=granite attempts=1 input_chars=4 "
        r"truncated=False prompt_tokens=11 completion_tokens=7",
        record.getMessage(),
    )


@pytest.mark.respx(assert_all_called=False)
async def test_log_line_unknown_fields_are_dashes(
    respx_mock: respx.MockRouter, serve: Serve, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    async with serve(create_app(_settings())) as client:
        await client.post("/classify", content=b"{}")
    (record,) = _info_records(caplog)
    assert re.fullmatch(
        r"classify outcome=invalid_input latency_ms=\d+ model=- attempts=0 input_chars=- "
        r"truncated=- prompt_tokens=- completion_tokens=-",
        record.getMessage(),
    )


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize(
    ("message", "runtime", "boom", "outcome"),
    [
        pytest.param("  ", None, False, "invalid_input", id="invalid_input"),
        pytest.param("x", httpx.ReadTimeout("slow"), False, "llm_timeout", id="llm_timeout"),
        pytest.param("x", None, True, "internal_error", id="internal_error"),
    ],
)
async def test_log_outcome_on_errors(
    respx_mock: respx.MockRouter,
    serve: Serve,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    message: str,
    runtime: Exception | None,
    boom: bool,
    outcome: str,
) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    respx_mock.post(URL).mock(side_effect=[runtime] if runtime else None, return_value=_good())
    if boom:
        _boom_classify(monkeypatch)
    async with serve(create_app(_settings())) as client:
        await _post(client, message)
    (record,) = _info_records(caplog)
    assert f"outcome={outcome} " in record.getMessage()


async def test_request_validation_error_handler(serve: Serve) -> None:
    app = create_app(_settings())

    async def needs(q: int) -> dict[str, int]:
        return {"q": q}

    app.add_api_route("/needs", needs)
    async with serve(app) as client:
        response = await client.get("/needs")
    assert response.status_code == 400
    assert response.json() == _error_body("invalid_input", "Invalid request: query.q: missing")


async def test_fallback_exception_handler() -> None:
    app = create_app(_settings())

    async def boom() -> None:
        raise RuntimeError("secret-marker")

    app.add_api_route("/boom", boom)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://api.test") as client,
    ):
        response = await client.get("/boom")
    assert response.status_code == 500
    assert response.json() == _error_body("internal_error", "Internal error")


async def test_log_latency_is_the_route_time_for_success(
    serve: Serve, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")

    async def claim_huge_latency(
        self: Classifier, message: str, stats: ClassifyStats
    ) -> Classification:
        result = ClassificationResult(
            category=Category.UNKNOWN,
            severity=Severity.LOW,
            summary="s",
            suggested_action="a",
        )
        return Classification(result, "granite", False, 987654)

    monkeypatch.setattr(Classifier, "classify", claim_huge_latency)
    async with serve(create_app(_settings())) as client:
        response = await _post(client, "boom")
    assert response.json()["latency_ms"] == 987654
    (record,) = _info_records(caplog)
    match = re.search(r"latency_ms=(\d+)", record.getMessage())
    assert match is not None
    assert int(match.group(1)) < 60_000
