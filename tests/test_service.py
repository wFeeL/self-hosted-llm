import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import pytest

from error_classifier.classify.prompt import (
    CUT_OFF_CORRECTION,
    INPUT_PREFIX,
    build_messages,
    correction_for,
    normalize_input,
)
from error_classifier.classify.schema import SCHEMA_NAME, Category, Severity, response_schema
from error_classifier.classify.service import (
    Classification,
    Classifier,
    ClassifierOptions,
    ClassifyStats,
)
from error_classifier.config import Settings
from error_classifier.errors import (
    InvalidInputError,
    LlmBadResponseError,
    LlmTimeoutError,
    LlmUnavailableError,
    ModelOutputInvalidError,
    ServiceError,
)
from error_classifier.llm.client import ChatCompletion, ChatMessage

pytestmark = pytest.mark.anyio

OPTIONS = ClassifierOptions(
    max_input_chars=3000,
    max_output_tokens=100,
    temperature=0.25,
    timeout_s=9.0,
    disable_thinking=True,
)
VALID: dict[str, str] = {
    "category": "network_error",
    "severity": "high",
    "summary": "The upstream refused the connection.",
    "suggested_action": "Check that the upstream is running.",
}


def _json(**changes: str) -> str:
    return json.dumps({**VALID, **changes})


VALID_JSON = _json()


def _ok(
    content: str | None = VALID_JSON,
    *,
    finish_reason: str = "stop",
    model: str = "granite",
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
) -> ChatCompletion:
    return ChatCompletion(content, finish_reason, model, prompt_tokens, completion_tokens)


class FakeChat:
    def __init__(self, *outcomes: ChatCompletion | Exception) -> None:
        self._outcomes = list(outcomes)
        self.messages: list[list[ChatMessage]] = []
        self.kwargs: list[dict[str, object]] = []

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
        self.messages.append(list(messages))
        self.kwargs.append(
            {
                "schema": schema,
                "schema_name": schema_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout_s": timeout_s,
                "disable_thinking": disable_thinking,
            }
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _clock(*values: float) -> Callable[[], float]:
    # Returns the given values in turn, then repeats the last one.
    ticks = iter(values)
    last = values[-1]

    def clock() -> float:
        nonlocal last
        last = next(ticks, last)
        return last

    return clock


def _classifier(
    fake: FakeChat,
    options: ClassifierOptions = OPTIONS,
    clock: Callable[[], float] | None = None,
) -> Classifier:
    return Classifier(fake, options, clock=clock or _clock(0.0, 1.234))


async def test_success_first_try() -> None:
    fake = FakeChat(_ok())
    stats = ClassifyStats()
    result = await _classifier(fake).classify("  boom  ", stats)

    assert fake.messages == [build_messages("boom")]
    assert fake.kwargs == [
        {
            "schema": response_schema(),
            "schema_name": SCHEMA_NAME,
            "max_tokens": 100,
            "temperature": 0.25,
            "timeout_s": 9.0,
            "disable_thinking": True,
        }
    ]
    assert isinstance(result, Classification)
    assert result.result.category is Category.NETWORK_ERROR
    assert result.result.severity is Severity.HIGH
    assert result.model == "granite"
    assert result.truncated is False
    assert result.latency_ms == 1234
    assert stats == ClassifyStats(
        attempts=1,
        input_chars=4,
        truncated=False,
        model="granite",
        prompt_tokens=10,
        completion_tokens=5,
    )


async def test_stats_filled_on_failure() -> None:
    fake = FakeChat(_ok("nope"), _ok("still nope", model="granite-2"))
    stats = ClassifyStats()
    with pytest.raises(ModelOutputInvalidError):
        await _classifier(fake).classify("boom", stats)
    assert stats == ClassifyStats(
        attempts=2,
        input_chars=4,
        truncated=False,
        model="granite-2",
        prompt_tokens=20,
        completion_tokens=10,
    )


@pytest.mark.parametrize(
    ("first", "reason", "echoes_answer"),
    [
        pytest.param(_ok("not json"), "answer: json_invalid", True, id="not-json"),
        pytest.param(_ok(_json(category="bogus")), "category: enum", True, id="foreign-enum"),
        pytest.param(
            _ok(_json(summary="   ")), "summary: string_too_short", True, id="blank-summary"
        ),
        pytest.param(
            _ok(_json(summary="x" * 301)), "summary: string_too_long", True, id="long-summary"
        ),
        pytest.param(
            _ok(json.dumps({**VALID, "extra": "bogus"})),
            "extra: extra_forbidden",
            True,
            id="extra-field",
        ),
        pytest.param(_ok(None), "empty answer", False, id="none"),
        pytest.param(_ok("  \n"), "empty answer", False, id="blank"),
    ],
)
async def test_retry_after_invalid(first: ChatCompletion, reason: str, echoes_answer: bool) -> None:
    fake = FakeChat(first, _ok())
    stats = ClassifyStats()
    result = await _classifier(fake).classify("boom", stats)

    base = build_messages("boom")
    correction: ChatMessage = {"role": "user", "content": correction_for([reason])}
    expected = [*base]
    if echoes_answer:
        assert first.content is not None
        expected.append({"role": "assistant", "content": first.content})
    expected.append(correction)
    assert fake.messages == [base, expected]
    assert "bogus" not in correction["content"]
    assert result.result.category is Category.NETWORK_ERROR
    assert stats.attempts == 2


async def test_retry_after_length() -> None:
    cut = _ok('{"category":"netw', finish_reason="length", completion_tokens=100)
    fake = FakeChat(cut, _ok())
    result = await _classifier(fake).classify("boom", ClassifyStats())

    base = build_messages("boom")
    assert fake.messages == [base, [*base, {"role": "user", "content": CUT_OFF_CORRECTION}]]
    assert result.result.severity is Severity.HIGH


async def test_retry_after_length_without_token_count() -> None:
    cut = _ok('{"category":"netw', finish_reason="length", completion_tokens=None)
    fake = FakeChat(cut, _ok())
    await _classifier(fake).classify("boom", ClassifyStats())
    assert len(fake.messages) == 2


async def test_length_below_token_limit_means_input_too_long() -> None:
    # The slot ran out before max_tokens did, so the prompt was too long for the context.
    fake = FakeChat(_ok('{"cat', finish_reason="length", completion_tokens=99), _ok())
    stats = ClassifyStats()
    with pytest.raises(InvalidInputError) as info:
        await _classifier(fake).classify("boom", stats)
    assert info.value.message == "The error text is too long for the model context, shorten it"
    assert len(fake.messages) == 1
    assert stats.attempts == 1


async def test_two_invalid_answers() -> None:
    fake = FakeChat(_ok("nope"), _ok(_json(severity="x")))
    with pytest.raises(ModelOutputInvalidError) as info:
        await _classifier(fake).classify("boom", ClassifyStats())
    assert info.value.message == "The model returned an invalid answer twice"
    assert len(fake.messages) == 2


async def test_two_cut_off_answers() -> None:
    cut = _ok("{", finish_reason="length", completion_tokens=100)
    fake = FakeChat(cut, cut)
    with pytest.raises(ModelOutputInvalidError):
        await _classifier(fake).classify("boom", ClassifyStats())
    assert len(fake.messages) == 2


async def test_unknown_finish_reason() -> None:
    fake = FakeChat(_ok(finish_reason="content_filter"))
    with pytest.raises(LlmBadResponseError) as info:
        await _classifier(fake).classify("boom", ClassifyStats())
    assert info.value.message == "LLM runtime finished with an unexpected reason"
    assert len(fake.messages) == 1


@pytest.mark.parametrize(
    "error",
    [
        LlmUnavailableError("down"),
        LlmBadResponseError("bad"),
        InvalidInputError("long"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_runtime_errors_not_retried(error: ServiceError) -> None:
    fake = FakeChat(error)
    stats = ClassifyStats()
    with pytest.raises(type(error)) as info:
        await _classifier(fake).classify("boom", stats)
    assert info.value is error
    assert len(fake.messages) == 1
    assert stats.attempts == 1


DEADLINE = replace(OPTIONS, timeout_s=120.0)
RETRY_TIMEOUT = (
    "The model gave an invalid answer and the 120 s request deadline left no time for a retry"
)


async def test_second_attempt_gets_the_remaining_time() -> None:
    fake = FakeChat(_ok("not json"), _ok())
    await _classifier(fake, DEADLINE, _clock(0.0, 40.0, 41.0)).classify("boom", ClassifyStats())
    assert [kw["timeout_s"] for kw in fake.kwargs] == [120.0, 80.0]


async def test_too_little_time_for_retry() -> None:
    fake = FakeChat(_ok("not json"), _ok())
    stats = ClassifyStats()
    with pytest.raises(LlmTimeoutError) as info:
        await _classifier(fake, DEADLINE, _clock(0.0, 116.0)).classify("boom", stats)
    assert info.value.message == RETRY_TIMEOUT
    assert len(fake.messages) == 1
    assert stats.attempts == 1


async def test_retry_with_exactly_min_time() -> None:
    fake = FakeChat(_ok("not json"), _ok())
    await _classifier(fake, DEADLINE, _clock(0.0, 115.0)).classify("boom", ClassifyStats())
    assert fake.kwargs[1]["timeout_s"] == 5.0


async def test_runtime_timeout_names_the_deadline() -> None:
    fake = FakeChat(LlmTimeoutError("x"))
    stats = ClassifyStats()
    with pytest.raises(LlmTimeoutError) as info:
        await _classifier(fake, DEADLINE).classify("boom", stats)
    assert info.value.message == "The model did not answer within the 120 s request deadline"
    assert len(fake.messages) == 1
    assert stats.attempts == 1


async def test_truncated_input_flag() -> None:
    fake = FakeChat(_ok())
    stats = ClassifyStats()
    options = ClassifierOptions(3000, 100, 0.0, 9.0, False)
    result = await _classifier(fake, options).classify("a" * 5000, stats)
    assert result.truncated is True
    assert stats.truncated is True
    assert stats.input_chars == 5000
    last = fake.messages[0][-1]["content"]
    assert "characters omitted" in last
    assert len(json.loads(last.removeprefix(INPUT_PREFIX))) <= 3000
    assert last == build_messages(normalize_input("a" * 5000, 3000).text)[-1]["content"]


async def test_tokens_summed_over_attempts() -> None:
    fake = FakeChat(_ok("nope"), _ok(prompt_tokens=7, completion_tokens=3))
    stats = ClassifyStats()
    await _classifier(fake).classify("boom", stats)
    assert (stats.prompt_tokens, stats.completion_tokens) == (17, 8)


async def test_tokens_sum_skips_missing_values() -> None:
    first = _ok("nope", prompt_tokens=None, completion_tokens=None)
    fake = FakeChat(first, _ok(prompt_tokens=7, completion_tokens=None))
    stats = ClassifyStats()
    await _classifier(fake).classify("boom", stats)
    assert (stats.prompt_tokens, stats.completion_tokens) == (7, None)


async def test_tokens_stay_none_when_runtime_gives_none() -> None:
    fake = FakeChat(_ok(prompt_tokens=None, completion_tokens=None))
    stats = ClassifyStats()
    await _classifier(fake).classify("boom", stats)
    assert (stats.prompt_tokens, stats.completion_tokens) == (None, None)


@pytest.mark.parametrize("fail", [False, True], ids=["success", "invalid-twice"])
async def test_service_logs_nothing_at_info(caplog: pytest.LogCaptureFixture, fail: bool) -> None:
    caplog.set_level(logging.DEBUG, logger="error_classifier")
    fake = FakeChat(_ok("SECRET-1"), _ok("SECRET-2") if fail else _ok())
    try:
        await _classifier(fake).classify("boom", ClassifyStats())
    except ModelOutputInvalidError:
        assert fail
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    assert any(r.levelno == logging.DEBUG for r in caplog.records)


def test_options_from_settings() -> None:
    settings = Settings(
        llm_model="m",
        request_timeout_s=33.0,
        max_input_chars=500,
        max_output_tokens=64,
        llm_temperature=0.7,
        llm_disable_thinking=True,
    )
    assert ClassifierOptions.from_settings(settings) == ClassifierOptions(
        max_input_chars=500,
        max_output_tokens=64,
        temperature=0.7,
        timeout_s=33.0,
        disable_thinking=True,
    )
