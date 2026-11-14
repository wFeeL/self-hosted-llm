import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from error_classifier.classify.service import (
    ChatClient,
    Classifier,
    ClassifierOptions,
    ClassifyStats,
)
from error_classifier.errors import LlmBadResponseError, LlmUnavailableError, ServiceError
from error_classifier.evaluation.dataset import EvalItem
from error_classifier.evaluation.metrics import AttemptRecord, ItemResult
from error_classifier.llm.client import ChatCompletion, ChatMessage

WARMUP_TEXT: Final = "ValueError: invalid literal for int() with base 10: 'abc'"
ENVIRONMENT_ERRORS: Final = (LlmUnavailableError, LlmBadResponseError)  # warm-up only
RECOVERY_WAIT_S: Final = 60.0
RECOVERY_STEP_S: Final = 5.0


class RecordingChat:
    """ChatClient wrapper that keeps the raw outcome of every attempt."""

    def __init__(self, inner: ChatClient) -> None:
        self._inner = inner
        self._records: list[AttemptRecord] = []

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
        try:
            completion = await self._inner.chat(
                messages,
                schema=schema,
                schema_name=schema_name,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                disable_thinking=disable_thinking,
            )
        except ServiceError as exc:
            self._records.append(
                AttemptRecord(
                    content=None,
                    finish_reason=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    error=exc.code,
                )
            )
            raise
        self._records.append(
            AttemptRecord(
                content=completion.content,
                finish_reason=completion.finish_reason,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                error=None,
            )
        )
        return completion

    def take(self) -> list[AttemptRecord]:
        records, self._records = self._records, []
        return records


StopReason = Literal["time_limit", "limit", "llm_unavailable", "llm_bad_response"]


@dataclass(frozen=True)
class RunOutcome:
    results: list[ItemResult]
    stop_reason: StopReason | None  # None means every item ran
    wall_s: float


async def _run_row(
    classifier: Classifier,
    recorder: RecordingChat,
    item: EvalItem,
    clock: Callable[[], float],
) -> ItemResult:
    """Classify one row; LlmUnavailableError is left to the caller."""
    stats = ClassifyStats()
    t0 = clock()
    try:
        classification = await classifier.classify(item.text, stats)
    except LlmUnavailableError:
        recorder.take()
        raise
    except ServiceError as exc:
        return ItemResult(
            id=item.id,
            outcome=exc.code,
            attempts=stats.attempts,
            latency_ms=round((clock() - t0) * 1000),
            truncated=bool(stats.truncated),
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            category=None,
            severity=None,
            summary=None,
            suggested_action=None,
            attempt_log=recorder.take(),
        )
    result = classification.result
    return ItemResult(
        id=item.id,
        outcome="ok",
        attempts=stats.attempts,
        latency_ms=classification.latency_ms,
        truncated=classification.truncated,
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
        category=result.category,
        severity=result.severity,
        summary=result.summary,
        suggested_action=result.suggested_action,
        attempt_log=recorder.take(),
    )


async def _wait_until_ready(
    probe: Callable[[], Awaitable[None]] | None,
    sleep: Callable[[float], Awaitable[None]],
) -> bool:
    if probe is None:
        return False
    for _ in range(int(RECOVERY_WAIT_S / RECOVERY_STEP_S)):
        await sleep(RECOVERY_STEP_S)
        try:
            await probe()
        except ServiceError:
            continue
        return True
    return False


async def run_eval(
    items: Sequence[EvalItem],
    chat: ChatClient,
    options: ClassifierOptions,
    *,
    max_s: float,
    limit: int | None = None,
    probe: Callable[[], Awaitable[None]] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.perf_counter,
    on_item: Callable[[ItemResult], None] | None = None,
) -> RunOutcome:
    recorder = RecordingChat(chat)
    classifier = Classifier(recorder, options, clock=clock)
    start = clock()
    try:
        await classifier.classify(WARMUP_TEXT, ClassifyStats())
    except ENVIRONMENT_ERRORS:
        raise
    except ServiceError:
        pass
    recorder.take()

    results: list[ItemResult] = []
    stop: StopReason | None = None
    for item in items:
        if limit is not None and len(results) >= limit:
            stop = "limit"
            break
        if clock() - start >= max_s:
            stop = "time_limit"
            break
        try:
            result = await _run_row(classifier, recorder, item, clock)
        except LlmUnavailableError:
            if not await _wait_until_ready(probe, sleep):
                stop = "llm_unavailable"
                break
            try:
                result = await _run_row(classifier, recorder, item, clock)
            except LlmUnavailableError:
                stop = "llm_unavailable"
                break
        results.append(result)
        if on_item is not None:
            on_item(result)
        if result.outcome == "llm_bad_response":
            if len(results) >= 2 and results[-2].outcome == "llm_bad_response":
                stop = "llm_bad_response"
                break
            if probe is not None:
                try:
                    await probe()
                except ServiceError:
                    stop = "llm_unavailable"
                    break
    return RunOutcome(results, stop, clock() - start)
