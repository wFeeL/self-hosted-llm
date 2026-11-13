import httpx
import pytest
import respx
from conftest import Serve
from fastapi import FastAPI

from error_classifier.api.app import create_app
from error_classifier.config import Settings

pytestmark = pytest.mark.anyio

URL = "http://llm.test/v1/models"


def _app() -> FastAPI:
    return create_app(Settings(llm_model="m", llm_base_url="http://llm.test/v1"))


async def test_ready_200(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.get(URL).respond(200, json={"data": [{"id": "m"}]})
    async with serve(_app()) as c:
        response = await c.get("/ready")
    assert route.call_count == 1
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": "m"}


async def test_ready_503_envelope_when_runtime_loading(
    respx_mock: respx.MockRouter, serve: Serve
) -> None:
    route = respx_mock.get(URL).respond(503)
    async with serve(_app()) as c:
        response = await c.get("/ready")
    assert route.call_count == 1
    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": "llm_unavailable", "message": "LLM runtime is still loading the model"}
    }


async def test_ready_503_when_alias_missing(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.get(URL).respond(200, json={"data": [{"id": "other"}]})
    async with serve(_app()) as c:
        response = await c.get("/ready")
    assert route.call_count == 1
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"


async def test_ready_503_when_unreachable(respx_mock: respx.MockRouter, serve: Serve) -> None:
    route = respx_mock.get(URL).mock(side_effect=httpx.ConnectError("refused"))
    async with serve(_app()) as c:
        response = await c.get("/ready")
    assert route.call_count == 1
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"
