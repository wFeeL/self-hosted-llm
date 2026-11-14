import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, Callable

import anyio
import httpx
import pytest
import respx
from conftest import Serve
from fakes import BlockingChat
from fastapi import FastAPI
from starlette.types import Message

from error_classifier.api.app import create_app
from error_classifier.classify.prompt import INPUT_PREFIX
from error_classifier.classify.schema import Category, ClassificationResult, Severity
from error_classifier.classify.service import (
    Classification,
    Classifier,
    ClassifierOptions,
    ClassifyStats,
)
from error_classifier.config import Settings
from error_classifier.errors import LlmTimeoutError, LlmUnavailableError
from error_classifier.llm.client import LlmClient

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
        r"classify outcome=ok total_ms=\d+ queue_ms=\d+ model=granite attempts=1 input_chars=4 "
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
        r"classify outcome=invalid_input total_ms=\d+ queue_ms=- model=- attempts=0 input_chars=- "
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
    match = re.search(r"total_ms=(\d+)", record.getMessage())
    assert match is not None
    assert int(match.group(1)) < 60_000


QUEUE_FULL_TEXT = "All model slots are busy and the queue is full, retry later"


def _blocking_app(**changes: object) -> FastAPI:
    return create_app(_settings(**{"max_concurrency": 1, "max_queue": 1, **changes}))


def _swap_in(app: FastAPI, chat: BlockingChat) -> None:
    app.state.classifier = Classifier(chat, ClassifierOptions.from_settings(app.state.settings))


async def _until(check: Callable[[], bool]) -> None:
    while not check():  # noqa: ASYNC110
        await asyncio.sleep(0.01)


def _request(client: httpx.AsyncClient, message: str = "boom") -> asyncio.Task[httpx.Response]:
    return asyncio.create_task(_post(client, message))


async def test_queue_full_gives_429(serve: Serve) -> None:
    chat = BlockingChat()
    app = _blocking_app()
    with anyio.fail_after(5):
        async with serve(app) as client:
            _swap_in(app, chat)
            first = _request(client)
            await chat.started.wait()
            second = _request(client)
            await _until(lambda: app.state.limiter.waiting == 1)
            third = await _post(client, "boom")
            chat.release.set()
            ok = [await first, await second]
    assert third.status_code == 429
    assert third.json() == _error_body("queue_full", QUEUE_FULL_TEXT)
    assert third.headers["retry-after"] == "30"
    assert [r.status_code for r in ok] == [200, 200]


async def test_queue_timeout_gives_429(serve: Serve) -> None:
    chat = BlockingChat()
    app = _blocking_app(queue_timeout_s=0.05)
    with anyio.fail_after(5):
        async with serve(app) as client:
            _swap_in(app, chat)
            first = _request(client)
            await chat.started.wait()
            second = await _post(client, "boom")
            chat.release.set()
            await first
    assert second.status_code == 429
    assert second.json() == _error_body(
        "queue_full", "No model slot became free within 0.05 s, retry later"
    )


async def test_slot_released_after_error(serve: Serve) -> None:
    chat = BlockingChat(errors=[LlmUnavailableError("down")])
    chat.release.set()
    app = _blocking_app()
    with anyio.fail_after(5):
        async with serve(app) as client:
            _swap_in(app, chat)
            failed = await _post(client, "boom")
            ok = await _post(client, "boom")
            active = app.state.limiter.active
    assert failed.status_code == 503
    assert ok.status_code == 200
    assert active == 0


async def test_invalid_input_skips_the_queue(serve: Serve) -> None:
    chat = BlockingChat()
    app = _blocking_app()
    with anyio.fail_after(5):
        async with serve(app) as client:
            _swap_in(app, chat)
            first = _request(client)
            await chat.started.wait()
            second = _request(client)
            await _until(lambda: app.state.limiter.waiting == 1)
            blank = await _post(client, "  ")
            chat.release.set()
            await first
            await second
    assert blank.status_code == 400


async def test_retry_after_header_rounds_up(serve: Serve) -> None:
    chat = BlockingChat()
    app = _blocking_app(max_queue=0, queue_timeout_s=0.2)
    with anyio.fail_after(5):
        async with serve(app) as client:
            _swap_in(app, chat)
            first = _request(client)
            await chat.started.wait()
            rejected = await _post(client, "boom")
            chat.release.set()
            await first
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "1"


async def test_log_line_has_queue_ms(serve: Serve, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    chat = BlockingChat()
    app = _blocking_app(max_queue=0)
    with anyio.fail_after(5):
        async with serve(app) as client:
            _swap_in(app, chat)
            first = _request(client)
            await chat.started.wait()
            await _post(client, "boom")
            await _post(client, "  ")
            chat.release.set()
            await first
    lines = [r.getMessage() for r in _info_records(caplog)]
    assert re.search(r"outcome=queue_full total_ms=\d+ queue_ms=\d+ ", lines[0])
    assert re.search(r"outcome=invalid_input total_ms=\d+ queue_ms=- ", lines[1])
    assert re.search(r"outcome=ok total_ms=\d+ queue_ms=\d+ ", lines[2])


async def _asgi_post(
    app: FastAPI, body: bytes, disconnect: asyncio.Event, *, body_complete: bool = True
) -> list[Message]:
    headers = [(b"content-type", b"application/json")]
    if body_complete:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/classify",
        "raw_path": b"/classify",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("api.test", 80),
    }
    sent: list[Message] = []
    body_sent = False

    async def receive() -> Message:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": not body_complete}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _status(messages: list[Message]) -> int:
    return next(m["status"] for m in messages if m["type"] == "http.response.start")


def _outcomes(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage().split()[1] for r in _info_records(caplog)]


BODY = b'{"message": "boom"}'


async def test_disconnect_cancels_the_runtime_call(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    chat = BlockingChat()
    app = _blocking_app()
    disconnect = asyncio.Event()
    with anyio.fail_after(5):
        async with app.router.lifespan_context(app):
            _swap_in(app, chat)
            task = asyncio.create_task(_asgi_post(app, BODY, disconnect))
            await chat.started.wait()
            disconnect.set()
            messages = await task
            active = app.state.limiter.active
    assert chat.cancelled == 1
    assert active == 0
    assert _outcomes(caplog) == ["outcome=client_closed"]
    assert _status(messages) == 499


async def test_disconnect_while_queued(serve: Serve, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    chat = BlockingChat()
    app = _blocking_app()
    disconnect = asyncio.Event()
    with anyio.fail_after(5):
        async with serve(app) as client:
            _swap_in(app, chat)
            first = _request(client)
            await chat.started.wait()
            second = asyncio.create_task(_asgi_post(app, BODY, disconnect))
            await _until(lambda: app.state.limiter.waiting == 1)
            disconnect.set()
            await second
            waiting = app.state.limiter.waiting
            chat.release.set()
            response = await first
    assert waiting == 0
    assert chat.calls == 1
    assert response.status_code == 200
    assert "outcome=client_closed" in _outcomes(caplog)


async def test_disconnect_cancels_a_real_client_call(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    started = asyncio.Event()

    async def slow(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(30)
        return httpx.Response(200)

    settings = _settings(request_timeout_s=60)
    app = create_app(settings)
    disconnect = asyncio.Event()
    with anyio.fail_after(5):
        async with (
            httpx.AsyncClient(
                transport=httpx.MockTransport(slow), base_url="http://llm.test/v1/"
            ) as http,
            app.router.lifespan_context(app),
        ):
            app.state.classifier = Classifier(
                LlmClient(http, "m", 2.0), ClassifierOptions.from_settings(settings)
            )
            task = asyncio.create_task(_asgi_post(app, BODY, disconnect))
            await started.wait()
            disconnected_at = time.perf_counter()
            disconnect.set()
            await task
            elapsed = time.perf_counter() - disconnected_at
    assert elapsed < 1
    assert _outcomes(caplog) == ["outcome=client_closed"]


async def test_disconnect_while_reading_body(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    app = create_app(_settings())
    disconnect = asyncio.Event()
    disconnect.set()
    with anyio.fail_after(5):
        async with app.router.lifespan_context(app):
            await _asgi_post(app, b'{"message": "bo', disconnect, body_complete=False)
    assert _outcomes(caplog) == ["outcome=client_closed"]
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


async def test_no_disconnect_returns_200() -> None:
    chat = BlockingChat()
    chat.release.set()
    app = _blocking_app()
    with anyio.fail_after(5):
        async with app.router.lifespan_context(app):
            _swap_in(app, chat)
            messages = await _asgi_post(app, BODY, asyncio.Event())
    assert _status(messages) == 200


async def test_no_disconnect_keeps_runtime_error_code() -> None:
    chat = BlockingChat(errors=[LlmTimeoutError("x")])
    app = _blocking_app()
    with anyio.fail_after(5):
        async with app.router.lifespan_context(app):
            _swap_in(app, chat)
            messages = await _asgi_post(app, BODY, asyncio.Event())
    assert _status(messages) == 504
