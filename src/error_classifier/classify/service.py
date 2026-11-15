import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

from pydantic import ValidationError

from error_classifier.classify.prompt import (
    CUT_OFF_CORRECTION,
    build_messages,
    correction_for,
    normalize_input,
)
from error_classifier.classify.redact import mask_emails
from error_classifier.classify.schema import SCHEMA_NAME, ClassificationResult, response_schema
from error_classifier.config import Settings
from error_classifier.errors import (
    InvalidInputError,
    LlmBadResponseError,
    LlmTimeoutError,
    ModelOutputInvalidError,
)
from error_classifier.llm.client import ChatCompletion, ChatMessage

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
MIN_RETRY_S = 5.0


class ChatClient(Protocol):
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
    ) -> ChatCompletion: ...


@dataclass(frozen=True)
class ClassifierOptions:
    max_input_chars: int
    max_output_tokens: int
    temperature: float
    timeout_s: float
    disable_thinking: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            max_input_chars=settings.max_input_chars,
            max_output_tokens=settings.max_output_tokens,
            temperature=settings.llm_temperature,
            timeout_s=settings.request_timeout_s,
            disable_thinking=settings.llm_disable_thinking,
        )


@dataclass
class ClassifyStats:
    # Filled while classify runs, so the route can log attempts and tokens on errors too.
    attempts: int = 0
    input_chars: int | None = None
    truncated: bool | None = None
    model: str | None = None
    prompt_tokens: int | None = None  # sum over attempts, None if the runtime gave none
    completion_tokens: int | None = None


@dataclass(frozen=True)
class Classification:
    result: ClassificationResult
    model: str
    truncated: bool
    latency_ms: int


def _add(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    return value if total is None else total + value


def _reasons(error: ValidationError) -> list[str]:
    # Field and error type only: the invalid value can echo the user's text.
    return [
        f"{'.'.join(map(str, err['loc'])) or 'answer'}: {err['type']}"
        for err in error.errors(include_input=False, include_url=False)
    ]


class Classifier:
    def __init__(
        self,
        llm: ChatClient,
        options: ClassifierOptions,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._llm = llm
        self._options = options
        self._clock = clock

    async def classify(self, message: str, stats: ClassifyStats) -> Classification:
        start = self._clock()
        norm = normalize_input(message, self._options.max_input_chars)
        stats.input_chars = norm.original_chars
        stats.truncated = norm.truncated
        base = build_messages(norm.text)
        messages: list[ChatMessage] = base
        total_s = self._options.timeout_s
        deadline = start + total_s
        for attempt in range(MAX_ATTEMPTS):
            call_s = total_s
            if attempt > 0:
                call_s = deadline - self._clock()
                if call_s < MIN_RETRY_S:
                    raise LlmTimeoutError(
                        f"The model gave an invalid answer and the {total_s:g} s request "
                        "deadline left no time for a retry"
                    )
            stats.attempts += 1
            try:
                completion = await self._llm.chat(
                    messages,
                    schema=response_schema(),
                    schema_name=SCHEMA_NAME,
                    max_tokens=self._options.max_output_tokens,
                    temperature=self._options.temperature,
                    timeout_s=call_s,
                    disable_thinking=self._options.disable_thinking,
                )
            except LlmTimeoutError as exc:
                raise LlmTimeoutError(
                    f"The model did not answer within the {total_s:g} s request deadline"
                ) from exc
            stats.model = completion.model
            stats.prompt_tokens = _add(stats.prompt_tokens, completion.prompt_tokens)
            stats.completion_tokens = _add(stats.completion_tokens, completion.completion_tokens)

            if completion.finish_reason == "length":
                tokens = completion.completion_tokens
                if tokens is not None and tokens < self._options.max_output_tokens:
                    # Stopped before max_tokens, so the context slot ran out, not the answer budget.
                    raise InvalidInputError(
                        "The error text is too long for the model context, shorten it"
                    )
                logger.debug("attempt %d was cut off at the token limit", stats.attempts)
                messages = [*base, {"role": "user", "content": CUT_OFF_CORRECTION}]
                continue
            if completion.finish_reason != "stop":
                raise LlmBadResponseError("LLM runtime finished with an unexpected reason")
            content = completion.content
            if content is None or not content.strip():
                logger.debug("attempt %d returned an empty answer", stats.attempts)
                correction = correction_for(["empty answer"])
                messages = [*base, {"role": "user", "content": correction}]
                continue
            try:
                result = ClassificationResult.model_validate_json(content)
            except ValidationError as exc:
                reasons = _reasons(exc)
                logger.debug("attempt %d invalid: %s; content=%r", stats.attempts, reasons, content)
                messages = [
                    *base,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": correction_for(reasons)},
                ]
                continue
            # The model may repeat an address from the error text;
            # the caller already has the original.
            result = result.model_copy(
                update={
                    "summary": mask_emails(result.summary),
                    "suggested_action": mask_emails(result.suggested_action),
                }
            )
            latency_ms = round((self._clock() - start) * 1000)
            return Classification(result, completion.model, norm.truncated, latency_ms)
        raise ModelOutputInvalidError("The model returned an invalid answer twice")
