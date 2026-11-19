import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import httpx

from error_classifier.config import Settings
from error_classifier.errors import (
    InvalidInputError,
    LlmBadResponseError,
    LlmTimeoutError,
    LlmUnavailableError,
)

_MISSING = object()


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatCompletion:
    content: str | None
    finish_reason: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    # Finite default so no call can hang; calls with their own needs override it per request.
    return httpx.AsyncClient(
        base_url=settings.llm_base_url + "/",
        timeout=httpx.Timeout(settings.llm_probe_timeout_s),
    )


def break_turn_tokens(text: str) -> str:
    # llama-server parses special tokens inside message text, so "<|im_end|>" from an
    # error or a model answer could close the turn; a JSON escape keeps the text readable.
    return text.replace("<|", "<\\u007c").replace("|>", "\\u007c>")


class LlmClient:
    def __init__(self, http: httpx.AsyncClient, model: str, probe_timeout_s: float) -> None:
        self._http = http
        self.model = model
        self.probe_timeout_s = probe_timeout_s

    async def check_ready(self) -> None:
        try:
            response = await self._http.get("models", timeout=self.probe_timeout_s)
        except httpx.TimeoutException as exc:
            raise LlmUnavailableError(
                f"LLM runtime did not answer within {self.probe_timeout_s} s"
            ) from exc
        except httpx.TransportError as exc:
            raise LlmUnavailableError("LLM runtime is unreachable") from exc
        except httpx.RequestError as exc:
            raise LlmUnavailableError("LLM runtime request failed") from exc
        if response.status_code == 503:
            raise LlmUnavailableError("LLM runtime is still loading the model")
        if response.status_code != 200:
            raise LlmUnavailableError(f"LLM runtime returned HTTP {response.status_code}")
        if self.model not in _model_ids(response):
            raise LlmUnavailableError(f"Model '{self.model}' is not served by the LLM runtime")

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, object],
        schema_name: str,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        disable_thinking: bool,
    ) -> ChatCompletion:
        body: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": m["role"], "content": break_turn_tokens(m["content"])} for m in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": dict(schema)},
            },
        }
        if disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            async with asyncio.timeout(timeout_s):
                response = await self._http.post("chat/completions", json=body, timeout=timeout_s)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise LlmTimeoutError(f"LLM runtime did not answer within {timeout_s:g} s") from exc
        except httpx.TransportError as exc:
            raise LlmUnavailableError("LLM runtime is unreachable") from exc
        except httpx.RequestError as exc:
            raise LlmUnavailableError("LLM runtime request failed") from exc
        _raise_for_status(response)
        return _parse_completion(response)


def _json_object(response: httpx.Response) -> dict[str, object] | None:
    try:
        body = cast(object, response.json())
    except ValueError:
        return None
    return cast(dict[str, object], body) if isinstance(body, dict) else None


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status == 200:
        return
    if status == 503:
        raise LlmUnavailableError("LLM runtime is still loading the model")
    if status >= 500:
        raise LlmUnavailableError(f"LLM runtime returned HTTP {status}")
    if 400 <= status < 500:
        body = _json_object(response)
        error = body.get("error") if body is not None else None
        kind = cast(dict[str, object], error).get("type") if isinstance(error, dict) else None
        if status == 400 and kind == "exceed_context_size_error":
            raise InvalidInputError("The error text is too long for the model context, shorten it")
        if not isinstance(kind, str) or re.fullmatch(r"[a-z_]{1,64}", kind) is None:
            kind = "unknown"
        raise LlmBadResponseError(f"LLM runtime rejected the request: HTTP {status} {kind}")
    raise LlmBadResponseError(f"LLM runtime returned HTTP {status}")


def _invalid(field: str) -> LlmBadResponseError:
    return LlmBadResponseError(f"LLM runtime response has no valid '{field}'")


def _token_count(usage: object, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = cast(dict[str, object], usage).get(key)
    # bool is an int subclass, but a flag is not a token count.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_completion(response: httpx.Response) -> ChatCompletion:
    body = _json_object(response)
    if body is None:
        raise _invalid("body")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _invalid("choices")
    choice = cast(list[object], choices)[0]
    if not isinstance(choice, dict):
        raise _invalid("choices")
    choice = cast(dict[str, object], choice)
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise _invalid("finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _invalid("message")
    message = cast(dict[str, object], message)
    content = message.get("content", _MISSING)
    if content is not None and not isinstance(content, str):
        raise _invalid("content")
    model = body.get("model")
    if not isinstance(model, str):
        raise _invalid("model")
    usage = body.get("usage")
    return ChatCompletion(
        content=content,
        finish_reason=finish_reason,
        model=model,
        prompt_tokens=_token_count(usage, "prompt_tokens"),
        completion_tokens=_token_count(usage, "completion_tokens"),
    )


def _model_ids(response: httpx.Response) -> list[str]:
    unexpected = LlmUnavailableError("LLM runtime returned an unexpected model list")
    try:
        body = cast(object, response.json())
    except ValueError as exc:
        raise unexpected from exc
    if not isinstance(body, dict):
        raise unexpected
    data = cast(dict[str, object], body).get("data")
    if not isinstance(data, list):
        raise unexpected
    ids: list[str] = []
    for item in cast(list[object], data):
        model_id = cast(dict[str, object], item).get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str):
            raise unexpected
        ids.append(model_id)
    return ids
