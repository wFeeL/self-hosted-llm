from typing import cast

from fastapi import APIRouter, Request

from error_classifier.llm.client import LlmClient

router = APIRouter()


def get_llm(request: Request) -> LlmClient:
    return cast(LlmClient, request.app.state.llm)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    llm = get_llm(request)
    await llm.check_ready()
    return {"status": "ready", "model": llm.model}
