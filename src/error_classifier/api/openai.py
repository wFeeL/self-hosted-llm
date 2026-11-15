from typing import cast

from fastapi import APIRouter, Request

from error_classifier.config import Settings
from error_classifier.errors import ModelNotFoundError

router = APIRouter(prefix="/v1")


def _alias(request: Request) -> str:
    return cast(Settings, request.app.state.settings).llm_model


def model_card(alias: str) -> dict[str, object]:
    return {"id": alias, "object": "model", "created": 0, "owned_by": "llm-error-classifier"}


@router.get("/models")
async def list_models(request: Request) -> dict[str, object]:
    return {"object": "list", "data": [model_card(_alias(request))]}


@router.get("/models/{model}")
async def retrieve_model(model: str, request: Request) -> dict[str, object]:
    alias = _alias(request)
    if model != alias:
        raise ModelNotFoundError(
            f"Model '{model}' is not served; available: {alias}", param="model"
        )
    return model_card(alias)
