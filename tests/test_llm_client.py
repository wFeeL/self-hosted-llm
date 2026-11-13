from collections.abc import AsyncGenerator

import httpx
import pytest
import respx

from error_classifier.config import Settings
from error_classifier.errors import LlmUnavailableError
from error_classifier.llm.client import LlmClient, build_http_client

pytestmark = pytest.mark.anyio

URL = "http://llm.test/v1/models"
UNEXPECTED = "LLM runtime returned an unexpected model list"
NOT_SERVED = "Model 'm' is not served by the LLM runtime"


@pytest.fixture
async def llm() -> AsyncGenerator[LlmClient]:
    settings = Settings(llm_model="m", llm_base_url="http://llm.test/v1")
    async with build_http_client(settings) as http:
        yield LlmClient(http, "m", 2.0)


async def test_ready_when_alias_listed(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    route = respx_mock.get(URL).respond(200, json={"data": [{"id": "other"}, {"id": "m"}]})
    await llm.check_ready()
    assert route.call_count == 1


async def test_timeout(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    respx_mock.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(LlmUnavailableError) as info:
        await llm.check_ready()
    assert info.value.message == "LLM runtime did not answer within 2.0 s"


async def test_connection_refused(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    respx_mock.get(URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(LlmUnavailableError) as info:
        await llm.check_ready()
    assert info.value.message == "LLM runtime is unreachable"


async def test_runtime_loading_503(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    respx_mock.get(URL).respond(503)
    with pytest.raises(LlmUnavailableError) as info:
        await llm.check_ready()
    assert info.value.message == "LLM runtime is still loading the model"


@pytest.mark.parametrize("status", [404, 401, 500])
async def test_other_status(respx_mock: respx.MockRouter, llm: LlmClient, status: int) -> None:
    respx_mock.get(URL).respond(status)
    with pytest.raises(LlmUnavailableError) as info:
        await llm.check_ready()
    assert info.value.message == f"LLM runtime returned HTTP {status}"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"<html>", id="not-json"),
        pytest.param(b"{}", id="no-data"),
        pytest.param(b'{"data": "x"}', id="data-not-list"),
        pytest.param(b'{"data": [{}]}', id="item-without-id"),
        pytest.param(b'{"data": [{"id": 1}]}', id="id-not-string"),
        pytest.param(b"[]", id="top-level-list"),
        pytest.param(b'{"data": [{"id": "m"}, 5]}', id="item-not-object"),
    ],
)
async def test_unexpected_body(respx_mock: respx.MockRouter, llm: LlmClient, body: bytes) -> None:
    respx_mock.get(URL).respond(200, content=body)
    with pytest.raises(LlmUnavailableError) as info:
        await llm.check_ready()
    assert info.value.message == UNEXPECTED


async def test_empty_model_list(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    respx_mock.get(URL).respond(200, json={"data": []})
    with pytest.raises(LlmUnavailableError) as info:
        await llm.check_ready()
    assert info.value.message == NOT_SERVED


async def test_alias_missing(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    respx_mock.get(URL).respond(200, json={"data": [{"id": "other"}]})
    with pytest.raises(LlmUnavailableError) as info:
        await llm.check_ready()
    assert info.value.message == NOT_SERVED


async def test_probe_request_carries_finite_timeouts(
    respx_mock: respx.MockRouter, llm: LlmClient
) -> None:
    route = respx_mock.get(URL).respond(200, json={"data": [{"id": "m"}]})
    await llm.check_ready()
    timeout = route.calls.last.request.extensions["timeout"]
    assert timeout == {"connect": 2.0, "read": 2.0, "write": 2.0, "pool": 2.0}


async def test_default_client_timeout_is_finite(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(URL).respond(200, json={})
    settings = Settings(llm_model="m", llm_base_url="http://llm.test/v1")
    async with build_http_client(settings) as http:
        await http.get("models")
    timeout = route.calls.last.request.extensions["timeout"]
    assert timeout == {"connect": 2.0, "read": 2.0, "write": 2.0, "pool": 2.0}


async def test_other_request_error(respx_mock: respx.MockRouter, llm: LlmClient) -> None:
    respx_mock.get(URL).mock(side_effect=httpx.DecodingError("x"))
    with pytest.raises(LlmUnavailableError) as info:
        await llm.check_ready()
    assert info.value.message == "LLM runtime request failed"
