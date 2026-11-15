import asyncio
import json
from collections.abc import Callable
from typing import Any, cast

import anyio
import httpx
import openai
import pytest
import respx
from conftest import Serve
from fakes import BlockingChat
from openai.types import Model
from openai.types.chat import ChatCompletion, ChatCompletionUserMessageParam

from error_classifier.api.app import create_app
from error_classifier.classify.schema import ClassificationResult
from error_classifier.classify.service import Classifier, ClassifierOptions
from error_classifier.config import Settings
from error_classifier.errors import ModelNotFoundError

pytestmark = pytest.mark.anyio

ALIAS = "granite"
CARD = {"id": ALIAS, "object": "model", "created": 0, "owned_by": "llm-error-classifier"}


URL = "http://llm.test/v1/chat/completions"
ANSWER = {
    "category": "network_error",
    "severity": "high",
    "summary": "The upstream refused the connection.",
    "suggested_action": "Check that the upstream is running.",
}


def _settings(**changes: object) -> Settings:
    values: dict[str, object] = {"llm_model": ALIAS, "llm_base_url": "http://llm.test/v1"}
    return Settings.model_validate({**values, **changes})


def _runtime(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "runtime-name",
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def _good() -> httpx.Response:
    return _runtime(json.dumps(ANSWER))


def _user(text: str) -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": text}


async def _create(client: httpx.AsyncClient, messages: Any, **extra: Any) -> ChatCompletion:
    create: Any = _sdk(client).chat.completions.create
    return cast(ChatCompletion, await create(model=ALIAS, messages=messages, **extra))


def _sent_to_runtime(route: respx.Route) -> str:
    return route.calls.last.request.content.decode()


def test_model_not_found_contract() -> None:
    error = ModelNotFoundError("nope", param="model")
    assert (error.code, error.status_code, error.param) == ("model_not_found", 404, "model")
    assert error.headers() == {}


async def test_models_lists_the_served_alias(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": [CARD]}


async def test_model_retrieve_returns_the_card(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        response = await client.get(f"/v1/models/{ALIAS}")
    assert response.status_code == 200
    assert response.json() == CARD


async def test_foreign_model_is_not_found(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        response = await client.get("/v1/models/gpt-4o")
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "model_not_found",
            "message": "Model 'gpt-4o' is not served; available: granite",
            "type": "invalid_request_error",
            "param": "model",
        }
    }


def _sdk(client: httpx.AsyncClient, **options: Any) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="http://api.test/v1", api_key="unused", http_client=cast(Any, client), **options
    )


async def test_sdk_models_pass_strict_validation(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        listed = await client.get("/v1/models")
        one = await client.get(f"/v1/models/{ALIAS}")
        page = await _sdk(client).models.list()
    for raw in (*listed.json()["data"], one.json()):
        assert Model.model_validate(raw, strict=True).id == ALIAS
    assert [m.id for m in page.data] == [ALIAS]


async def test_sdk_retrieve_foreign_model_is_not_found(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        with pytest.raises(openai.NotFoundError) as caught:
            await _sdk(client).models.retrieve("gpt-4o")
    assert caught.value.code == "model_not_found"
    assert caught.value.param == "model"


async def test_sdk_classifies_an_error(respx_mock: respx.MockRouter, serve: Serve) -> None:
    respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings())) as client:
        raw = await _sdk(client).chat.completions.with_raw_response.create(
            model=ALIAS, messages=[_user("ECONNREFUSED 10.0.0.5:5432")]
        )
    body = raw.http_response.json()
    completion = ChatCompletion.model_validate(body, strict=True)
    assert completion.model == ALIAS
    assert completion.id.startswith("chatcmpl-")
    content = completion.choices[0].message.content
    assert content is not None
    assert ClassificationResult.model_validate_json(content).model_dump(mode="json") == ANSWER
    assert list(json.loads(content)) == list(ANSWER)
    assert completion.usage is not None
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (11, 7)
    assert completion.usage.total_tokens == 18
    assert body["truncated"] is False
    assert isinstance(body["latency_ms"], int)


async def test_usage_is_left_out_when_the_runtime_gave_no_counts(
    respx_mock: respx.MockRouter, serve: Serve
) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ANSWER)}}],
            },
        )
    )
    async with serve(create_app(_settings())) as client:
        response = await client.post(
            "/v1/chat/completions", json={"model": ALIAS, "messages": [_user("x")]}
        )
    assert response.status_code == 200
    assert "usage" not in response.json()


async def test_missing_required_field_fails_validation(
    respx_mock: respx.MockRouter, serve: Serve
) -> None:
    respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings())) as client:
        response = await client.post(
            "/v1/chat/completions", json={"model": ALIAS, "messages": [_user("x")]}
        )
    body = response.json()
    ChatCompletion.model_validate(body, strict=True)
    del body["choices"][0]["index"]
    with pytest.raises(ValueError, match="index"):
        ChatCompletion.model_validate(body, strict=True)


async def test_wrong_model_is_not_found(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        with pytest.raises(openai.NotFoundError) as caught:
            await _sdk(client).chat.completions.create(model="gpt-4o", messages=[_user("x")])
    assert caught.value.code == "model_not_found"
    assert caught.value.param == "model"


TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "x", "schema": {"type": "object"}},
}


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("stream", {"stream": True}),
        ("n", {"n": 2}),
        ("tools", {"tools": TOOLS}),
        ("tool_choice", {"tool_choice": "required"}),
        ("logprobs", {"logprobs": True}),
        ("response_format", {"response_format": JSON_SCHEMA}),
        ("modalities", {"modalities": ["audio"]}),
        ("top_logprobs", {"logprobs": False, "top_logprobs": 3}),
    ],
)
async def test_rejected_parameters(serve: Serve, name: str, extra: dict[str, Any]) -> None:
    async with serve(create_app(_settings())) as client:
        with pytest.raises(openai.BadRequestError) as caught:
            await _create(client, [_user("x")], **extra)
    assert caught.value.param == name
    assert caught.value.code == "invalid_input"
    assert caught.value.message.endswith(f"Parameter '{name}' is not supported")


@pytest.mark.parametrize(
    "extra",
    [
        {"extra_body": {"tools": None, "top_logprobs": None}},
        {"tools": []},
        {"logprobs": False},
        {"modalities": ["text"]},
        {"temperature": 0.7},
        {"max_tokens": 10},
        {"response_format": {"type": "json_object"}},
        {"extra_body": {"unknown_key": 1}},
    ],
)
async def test_harmless_parameters_are_accepted(
    respx_mock: respx.MockRouter, serve: Serve, extra: dict[str, Any]
) -> None:
    respx_mock.post(URL).mock(return_value=_good())
    async with serve(create_app(_settings())) as client:
        completion = await _create(client, [_user("x")], **extra)
    assert completion.choices[0].finish_reason == "stop"


async def _bad_request(client: httpx.AsyncClient, body: object) -> httpx.Response:
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    return response


@pytest.mark.respx(assert_all_called=False)
async def test_last_message_must_be_user(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        with pytest.raises(openai.BadRequestError) as caught:
            await _create(client, [_user("x"), {"role": "assistant", "content": "y"}])
    assert caught.value.param == "messages"
    assert caught.value.message.endswith("The last message must have the role 'user'")


@pytest.mark.parametrize("content", ["", "  \n", []])
async def test_empty_user_message(serve: Serve, content: object) -> None:
    async with serve(create_app(_settings())) as client:
        response = await _bad_request(
            client,
            {"model": ALIAS, "messages": [{"role": "user", "content": content}]},
        )
    error = response.json()["error"]
    assert error["message"] == "The last user message must not be empty"
    assert error["param"] == "messages"


async def test_text_parts_are_joined(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.post(URL).mock(return_value=_good())
    parts = [{"type": "text", "text": "first line"}, {"type": "text", "text": "second line"}]
    async with serve(create_app(_settings())) as client:
        await _create(client, [{"role": "user", "content": parts}])
    sent = json.loads(_sent_to_runtime(route))
    # The prompt embeds the text as a JSON string, so the newline is escaped there.
    assert json.dumps("first line\nsecond line") in sent["messages"][-1]["content"]


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}],
        [{"type": "text"}],
        None,
        7,
    ],
)
async def test_non_text_content_is_rejected(serve: Serve, content: object) -> None:
    async with serve(create_app(_settings())) as client:
        response = await _bad_request(
            client, {"model": ALIAS, "messages": [{"role": "user", "content": content}]}
        )
    error = response.json()["error"]
    assert error["message"] == "Only text content parts are supported"
    assert error["param"] == "messages"


async def test_response_format_without_type_is_rejected(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        response = await _bad_request(
            client, {"model": ALIAS, "messages": [_user("x")], "response_format": {}}
        )
    assert response.json()["error"]["param"] == "response_format"


@pytest.mark.parametrize(
    "body",
    [
        {"model": ALIAS, "messages": []},
        {"model": ALIAS},
        {"model": ALIAS, "messages": [_user("x")], "tools": "oops"},
        {"model": ALIAS, "messages": [{"role": "wizard", "content": "x"}]},
    ],
)
async def test_schema_errors_are_invalid_input(serve: Serve, body: object) -> None:
    async with serve(create_app(_settings())) as client:
        response = await _bad_request(client, body)
    error = response.json()["error"]
    assert error["code"] == "invalid_input"
    assert error["message"].startswith("Invalid request: ")
    assert error["param"] is None


async def test_invalid_json_is_invalid_input(serve: Serve) -> None:
    async with serve(create_app(_settings())) as client:
        response = await client.post("/v1/chat/completions", content=b"{nope")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_input"


async def test_history_is_ignored(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.post(URL).mock(return_value=_good())
    messages = [
        {"role": "system", "content": "SYSTEM-MARKER"},
        _user("EARLIER-MARKER"),
        {"role": "assistant", "content": "ASSISTANT-MARKER"},
        _user("the real error"),
    ]
    async with serve(create_app(_settings())) as client:
        await _create(client, messages)
    sent = _sent_to_runtime(route)
    assert "the real error" in sent
    for marker in ("SYSTEM-MARKER", "EARLIER-MARKER", "ASSISTANT-MARKER"):
        assert marker not in sent


async def _until(check: Callable[[], bool]) -> None:
    while not check():  # noqa: ASYNC110
        await asyncio.sleep(0.01)


async def test_queue_full_is_a_rate_limit(serve: Serve) -> None:
    chat = BlockingChat()
    settings = _settings(max_concurrency=1, max_queue=1)
    app = create_app(settings)
    with anyio.fail_after(5):
        async with serve(app) as client:
            app.state.classifier = Classifier(chat, ClassifierOptions.from_settings(settings))
            sdk = _sdk(client, max_retries=0)

            def call() -> asyncio.Task[ChatCompletion]:
                return asyncio.create_task(
                    sdk.chat.completions.create(model=ALIAS, messages=[_user("x")])
                )

            first = call()
            await chat.started.wait()
            second = call()
            await _until(lambda: app.state.limiter.waiting == 1)
            with pytest.raises(openai.RateLimitError) as caught:
                await call()
            chat.release.set()
            await first
            await second
    assert caught.value.code == "queue_full"
    assert caught.value.response.headers["retry-after"] == "30"


async def test_invalid_model_output_is_not_retried_by_the_sdk(
    respx_mock: respx.MockRouter, serve: Serve
) -> None:
    route = respx_mock.post(URL).mock(side_effect=[_runtime("nope"), _runtime("nope")])
    async with serve(create_app(_settings())) as client:
        with pytest.raises(openai.InternalServerError) as caught:
            await _sdk(client).chat.completions.create(model=ALIAS, messages=[_user("x")])
    assert caught.value.code == "model_output_invalid"
    assert route.call_count == 2
