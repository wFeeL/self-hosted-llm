import pytest
from conftest import Serve

from error_classifier.api.app import create_app
from error_classifier.config import Settings
from error_classifier.errors import ModelNotFoundError

pytestmark = pytest.mark.anyio

ALIAS = "granite"
CARD = {"id": ALIAS, "object": "model", "created": 0, "owned_by": "llm-error-classifier"}


def _settings() -> Settings:
    return Settings(llm_model=ALIAS, llm_base_url="http://llm.test/v1")


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
