import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from error_classifier.classify.service import Classifier, ClassifyStats
from error_classifier.config import Settings
from error_classifier.errors import InvalidInputError, ServiceError
from error_classifier.llm.client import LlmClient

router = APIRouter()
logger = logging.getLogger("error_classifier.classify")


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str


def get_llm(request: Request) -> LlmClient:
    return cast(LlmClient, request.app.state.llm)


def get_classifier(request: Request) -> Classifier:
    return cast(Classifier, request.app.state.classifier)


def describe_invalid(errors: Iterable[Mapping[str, object]]) -> str:
    # Location and error type only: the rejected value may be the user's text.
    parts = [
        f"{'.'.join(map(str, cast(Sequence[object], err['loc']))) or 'body'}: {err['type']}"
        for err in errors
    ]
    return "Invalid request: " + "; ".join(parts)


async def read_body_limited(request: Request, limit: int) -> bytes:
    too_large = InvalidInputError(f"Request body is larger than {limit} bytes")
    declared = request.headers.get("content-length", "")
    if declared.isdecimal() and int(declared) > limit:
        raise too_large
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


def _dash(value: object) -> str:
    return "-" if value is None else str(value)


def _log_outcome(
    outcome: str, latency_ms: int, stats: ClassifyStats, input_chars: int | None
) -> None:
    logger.info(
        "classify outcome=%s latency_ms=%d model=%s attempts=%s input_chars=%s truncated=%s "
        "prompt_tokens=%s completion_tokens=%s",
        outcome,
        latency_ms,
        _dash(stats.model),
        stats.attempts,
        _dash(input_chars),
        _dash(stats.truncated),
        _dash(stats.prompt_tokens),
        _dash(stats.completion_tokens),
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    llm = get_llm(request)
    await llm.check_ready()
    return {"status": "ready", "model": llm.model}


@router.post("/classify", response_model=None)
async def classify(request: Request) -> dict[str, object] | JSONResponse:
    started = time.perf_counter()
    settings = cast(Settings, request.app.state.settings)
    stats = ClassifyStats()
    message_chars: int | None = None

    def elapsed_ms() -> int:
        return round((time.perf_counter() - started) * 1000)

    try:
        body = await read_body_limited(request, settings.max_request_bytes)
        try:
            parsed = ClassifyRequest.model_validate_json(body)
        except ValidationError as exc:
            message = describe_invalid(exc.errors(include_input=False, include_url=False))
            raise InvalidInputError(message) from None
        message_chars = len(parsed.message.strip())
        if message_chars == 0:
            raise InvalidInputError("Field 'message' must not be empty")
        result = await get_classifier(request).classify(parsed.message, stats)
    except ServiceError as exc:
        _log_outcome(exc.code, elapsed_ms(), stats, stats.input_chars or message_chars)
        raise
    except Exception as exc:
        # Starlette re-raises after its own 500 handler, so the route answers itself.
        logger.error("classify failed with %s", type(exc).__name__)
        logger.debug("classify failure trace", exc_info=exc)
        _log_outcome("internal_error", elapsed_ms(), stats, stats.input_chars or message_chars)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal error"}},
        )
    _log_outcome("ok", result.latency_ms, stats, stats.input_chars)
    return {
        "category": result.result.category.value,
        "severity": result.result.severity.value,
        "summary": result.result.summary,
        "suggested_action": result.result.suggested_action,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "truncated": result.truncated,
    }
