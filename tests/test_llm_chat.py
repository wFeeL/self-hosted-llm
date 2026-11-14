import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import cast

import httpx
import pytest
import respx

from error_classifier.classify.schema import SCHEMA_NAME, response_schema
from error_classifier.config import Settings
from error_classifier.errors import (
    InvalidInputError,
    LlmBadResponseError,
    LlmTimeoutError,
    LlmUnavailableError,
    ServiceError,
)
from error_classifier.llm.client import ChatCompletion, ChatMessage, LlmClient, build_http_client

pytestmark = pytest.mark.anyio

URL = "http://llm.test/v1/chat/completions"
MESSAGES: list[ChatMessage] = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u"},
]


@pytest.fixture
async def llm() -> AsyncGenerator[LlmClient]:
    settings = Settings(llm_model="m", llm_base_url="http://llm.test/v1")
    async with build_http_client(settings) as http:
        yield LlmClient(http, "m", 2.0)


def _body(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "granite",
        "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    return {**body, **changes}


async def _chat(llm: LlmClient, *, disable_thinking: bool = False) -> ChatCompletion:
    return await llm.chat(
        MESSAGES,
        schema=response_schema(),
        schema_name=SCHEMA_NAME,
        max_tokens=100,
        temperature=0.5,
        timeout_s=7.0,
        disable_thinking=disable_thinking,
    )


async def test_chat_request_body(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    route = respx_mock.post(URL).respond(200, json=_body())
    await _chat(llm)
    request = route.calls.last.request
    assert json.loads(request.content) == {
        "model": "m",
        "messages": MESSAGES,
        "temperature": 0.5,
        "max_tokens": 100,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "classification", "schema": response_schema()},
        },
    }
    assert request.extensions["timeout"] == {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}


async def test_chat_disable_thinking_adds_template_kwargs(
    respx_mock: respx.MockRouter, llm: LlmClient
) -> None:
    route = respx_mock.post(URL).respond(200, json=_body())
    await _chat(llm, disable_thinking=True)
    sent = cast(dict[str, object], json.loads(route.calls.last.request.content))
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


async def test_chat_parses_completion(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    respx_mock.post(URL).respond(
        200,
        json=_body(
            choices=[
                {
                    "finish_reason": "length",
                    "message": {"content": "abc", "reasoning_content": "secret"},
                }
            ]
        ),
    )
    assert await _chat(llm) == ChatCompletion("abc", "length", "granite", 11, 7)


async def test_chat_null_content(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    respx_mock.post(URL).respond(
        200, json=_body(choices=[{"finish_reason": "stop", "message": {"content": None}}])
    )
    assert (await _chat(llm)).content is None


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(None, id="absent"),
        pytest.param("x", id="not-object"),
        pytest.param({"prompt_tokens": True, "completion_tokens": "7"}, id="bad-types"),
    ],
)
async def test_chat_missing_usage(
    respx_mock: respx.MockRouter, llm: LlmClient, usage: object
) -> None:
    body = _body()
    if usage is None:
        del body["usage"]
    else:
        body["usage"] = usage
    respx_mock.post(URL).respond(200, json=body)
    result = await _chat(llm)
    assert (result.prompt_tokens, result.completion_tokens) == (None, None)


def _error(status: int, body: object = None) -> httpx.Response:
    return httpx.Response(status, json=body) if body is not None else httpx.Response(status)


def _shape(**changes: object) -> httpx.Response:
    return httpx.Response(200, json=_body(**changes))


def _ok_choice(**message: object) -> list[object]:
    return [{"finish_reason": "stop", "message": message}]


def _no_key(key: str) -> httpx.Response:
    body = _body()
    del body[key]
    return httpx.Response(200, json=body)


FAILURES = [
    pytest.param(
        httpx.ReadTimeout("slow"),
        LlmTimeoutError,
        "LLM runtime did not answer within 7 s",
        id="timeout",
    ),
    pytest.param(
        httpx.ConnectError("refused"),
        LlmUnavailableError,
        "LLM runtime is unreachable",
        id="connect",
    ),
    pytest.param(
        httpx.DecodingError("x"), LlmUnavailableError, "LLM runtime request failed", id="decoding"
    ),
    pytest.param(
        _error(503), LlmUnavailableError, "LLM runtime is still loading the model", id="503"
    ),
    pytest.param(_error(500), LlmUnavailableError, "LLM runtime returned HTTP 500", id="500"),
    pytest.param(
        _error(400, {"error": {"type": "exceed_context_size_error", "message": "SECRET"}}),
        InvalidInputError,
        "The error text is too long for the model context, shorten it",
        id="context-overflow",
    ),
    pytest.param(
        _error(400, {"error": {"type": "invalid_request_error", "message": "SECRET"}}),
        LlmBadResponseError,
        "LLM runtime rejected the request: HTTP 400 invalid_request_error",
        id="400-other",
    ),
    pytest.param(
        _error(404),
        LlmBadResponseError,
        "LLM runtime rejected the request: HTTP 404 unknown",
        id="404-no-body",
    ),
    pytest.param(
        _error(422, {"error": "SECRET"}),
        LlmBadResponseError,
        "LLM runtime rejected the request: HTTP 422 unknown",
        id="error-not-object",
    ),
    pytest.param(
        _error(400, {"error": {"type": "Bad Type!", "message": "SECRET"}}),
        LlmBadResponseError,
        "LLM runtime rejected the request: HTTP 400 unknown",
        id="type-not-matching",
    ),
    pytest.param(
        _error(302), LlmBadResponseError, "LLM runtime returned HTTP 302", id="other-status"
    ),
    pytest.param(
        httpx.Response(200, content=b"<html>"),
        LlmBadResponseError,
        "LLM runtime response has no valid 'body'",
        id="body-not-json",
    ),
    pytest.param(
        httpx.Response(200, json=[]),
        LlmBadResponseError,
        "LLM runtime response has no valid 'body'",
        id="body-not-object",
    ),
    pytest.param(
        _no_key("choices"),
        LlmBadResponseError,
        "LLM runtime response has no valid 'choices'",
        id="no-choices",
    ),
    pytest.param(
        _shape(choices=[]),
        LlmBadResponseError,
        "LLM runtime response has no valid 'choices'",
        id="empty-choices",
    ),
    pytest.param(
        _shape(choices=["x"]),
        LlmBadResponseError,
        "LLM runtime response has no valid 'choices'",
        id="choice-not-object",
    ),
    pytest.param(
        _shape(choices=[{"message": {"content": "x"}}]),
        LlmBadResponseError,
        "LLM runtime response has no valid 'finish_reason'",
        id="no-finish-reason",
    ),
    pytest.param(
        _shape(choices=[{"finish_reason": "stop"}]),
        LlmBadResponseError,
        "LLM runtime response has no valid 'message'",
        id="no-message",
    ),
    pytest.param(
        _shape(choices=_ok_choice()),
        LlmBadResponseError,
        "LLM runtime response has no valid 'content'",
        id="no-content-key",
    ),
    pytest.param(
        _shape(choices=_ok_choice(content=5)),
        LlmBadResponseError,
        "LLM runtime response has no valid 'content'",
        id="content-not-string",
    ),
    pytest.param(
        _no_key("model"),
        LlmBadResponseError,
        "LLM runtime response has no valid 'model'",
        id="no-model",
    ),
]


@pytest.mark.parametrize(("outcome", "error", "message"), FAILURES)
async def test_chat_failures(
    respx_mock: respx.MockRouter,
    llm: LlmClient,
    outcome: httpx.Response | Exception,
    error: type[ServiceError],
    message: str,
) -> None:
    route = respx_mock.post(URL)
    if isinstance(outcome, Exception):
        route.mock(side_effect=outcome)
    else:
        route.mock(return_value=outcome)
    with pytest.raises(error) as info:
        await _chat(llm)
    assert type(info.value) is error
    assert info.value.message == message
    assert "SECRET" not in info.value.message


async def _chat_with_sleeping_transport(timeout_s: float) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://llm.test/v1/"
    ) as http:
        await LlmClient(http, "m", 2.0).chat(
            MESSAGES,
            schema=response_schema(),
            schema_name=SCHEMA_NAME,
            max_tokens=100,
            temperature=0.5,
            timeout_s=timeout_s,
            disable_thinking=False,
        )


async def test_chat_deadline_covers_the_whole_call() -> None:
    started = time.perf_counter()
    with pytest.raises(LlmTimeoutError):
        await _chat_with_sleeping_transport(0.2)
    assert time.perf_counter() - started < 0.7


async def test_timeout_message_uses_short_number() -> None:
    with pytest.raises(LlmTimeoutError) as info:
        await _chat_with_sleeping_transport(0.2)
    assert info.value.message == "LLM runtime did not answer within 0.2 s"
