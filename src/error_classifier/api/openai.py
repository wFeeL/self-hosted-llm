import json
import time
import uuid
from collections.abc import Callable
from functools import partial
from typing import Literal, cast

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from error_classifier.api.routes import classify_message, describe_invalid
from error_classifier.classify.service import Classification, ClassifyStats
from error_classifier.config import Settings
from error_classifier.errors import InvalidInputError, ModelNotFoundError

router = APIRouter(prefix="/v1")

TEXT_ONLY = "Only text content parts are supported"


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: object = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[OpenAIMessage] = Field(min_length=1)
    stream: bool | None = None
    n: int | None = None
    tools: list[object] | None = None
    tool_choice: object | None = None
    functions: list[object] | None = None
    function_call: object | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: dict[str, object] | None = None
    audio: object | None = None
    modalities: list[str] | None = None


def _alias(request: Request) -> str:
    return cast(Settings, request.app.state.settings).llm_model


def model_card(alias: str) -> dict[str, object]:
    return {"id": alias, "object": "model", "created": 0, "owned_by": "llm-error-classifier"}


def _unsupported(name: str) -> InvalidInputError:
    return InvalidInputError(f"Parameter '{name}' is not supported", param=name)


def _check_parameters(req: ChatCompletionRequest) -> None:
    # Only a present, non-null, non-empty value is refused, so wrappers sending nulls pass.
    rejected: list[tuple[str, bool]] = [
        ("stream", req.stream is True),
        ("n", req.n not in (None, 1)),
        ("tools", bool(req.tools)),
        ("tool_choice", req.tool_choice not in (None, "none")),
        ("functions", bool(req.functions)),
        ("function_call", req.function_call is not None),
        ("logprobs", req.logprobs is True),
        ("top_logprobs", req.top_logprobs is not None),
        (
            "response_format",
            req.response_format is not None
            and req.response_format.get("type") not in ("text", "json_object"),
        ),
        ("audio", req.audio is not None),
        ("modalities", any(m != "text" for m in req.modalities or [])),
    ]
    for name, refused in rejected:
        if refused:
            raise _unsupported(name)


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise InvalidInputError(TEXT_ONLY, param="messages")
    texts: list[str] = []
    for part in cast(list[object], content):
        if not isinstance(part, dict):
            raise InvalidInputError(TEXT_ONLY, param="messages")
        fields = cast(dict[str, object], part)
        text = fields.get("text")
        if fields.get("type") != "text" or not isinstance(text, str):
            raise InvalidInputError(TEXT_ONLY, param="messages")
        texts.append(text)
    return "\n".join(texts)


def message_from(req: ChatCompletionRequest, alias: str) -> str:
    if req.model != alias:
        raise model_not_found(req.model, alias)
    _check_parameters(req)
    last = req.messages[-1]
    if last.role != "user":
        raise InvalidInputError("The last message must have the role 'user'", param="messages")
    text = _text_of(last.content)
    if not text.strip():
        raise InvalidInputError("The last user message must not be empty", param="messages")
    return text


def parse_chat_request(body: bytes, alias: str) -> str:
    try:
        req = ChatCompletionRequest.model_validate_json(body)
    except ValidationError as exc:
        message = describe_invalid(exc.errors(include_input=False, include_url=False))
        raise InvalidInputError(message) from None
    return message_from(req, alias)


def model_not_found(model: str, alias: str) -> ModelNotFoundError:
    return ModelNotFoundError(f"Model '{model}' is not served; available: {alias}", param="model")


def completion_body(
    result: Classification,
    stats: ClassifyStats,
    alias: str,
    now: Callable[[], float] = time.time,
) -> dict[str, object]:
    answer = result.result
    content = json.dumps(
        {
            "category": answer.category.value,
            "severity": answer.severity.value,
            "summary": answer.summary,
            "suggested_action": answer.suggested_action,
        },
        ensure_ascii=False,
    )
    body: dict[str, object] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(now()),
        "model": alias,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "truncated": result.truncated,
        "latency_ms": result.latency_ms,
    }
    if stats.prompt_tokens is not None and stats.completion_tokens is not None:
        body["usage"] = {
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "total_tokens": stats.prompt_tokens + stats.completion_tokens,
        }
    return body


@router.get("/models")
async def list_models(request: Request) -> dict[str, object]:
    return {"object": "list", "data": [model_card(_alias(request))]}


@router.get("/models/{model}")
async def retrieve_model(model: str, request: Request) -> dict[str, object]:
    alias = _alias(request)
    if model != alias:
        raise model_not_found(model, alias)
    return model_card(alias)


@router.post("/chat/completions", response_model=None)
async def chat_completions(request: Request) -> dict[str, object] | Response:
    alias = _alias(request)
    outcome = await classify_message(request, partial(parse_chat_request, alias=alias), "chat")
    if isinstance(outcome, Response):
        return outcome
    result, stats = outcome
    return completion_body(result, stats, alias)
