import json
from collections.abc import Mapping, Sequence

import pytest

from error_classifier.classify.prompt import INPUT_PREFIX, build_messages
from error_classifier.classify.schema import Category, Severity
from error_classifier.classify.service import ClassifierOptions
from error_classifier.errors import (
    LlmBadResponseError,
    LlmTimeoutError,
    LlmUnavailableError,
    ModelOutputInvalidError,
)
from error_classifier.evaluation.dataset import EvalItem
from error_classifier.evaluation.metrics import AttemptRecord, ItemResult
from error_classifier.evaluation.runner import WARMUP_TEXT, RecordingChat, run_eval
from error_classifier.llm.client import ChatCompletion, ChatMessage

pytestmark = pytest.mark.anyio

GOOD = ChatCompletion(
    json.dumps(
        {
            "category": "runtime_error",
            "severity": "medium",
            "summary": "The code failed.",
            "suggested_action": "Check the code.",
        }
    ),
    "stop",
    "m-test",
    10,
    5,
)


class _Inner:
    def __init__(self, *replies: ChatCompletion | Exception) -> None:
        self.replies = list(replies)

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
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


async def _call(chat: RecordingChat) -> ChatCompletion:
    return await chat.chat(
        build_messages("x"),
        schema={},
        schema_name="s",
        max_tokens=16,
        temperature=0.0,
        timeout_s=1.0,
        disable_thinking=False,
    )


async def test_recording_chat_records_success_and_take_clears() -> None:
    recorder = RecordingChat(_Inner(GOOD))
    assert await _call(recorder) is GOOD
    assert recorder.take() == [
        AttemptRecord(
            content=GOOD.content,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            error=None,
        )
    ]
    assert recorder.take() == []


async def test_recording_chat_records_service_error_and_reraises() -> None:
    recorder = RecordingChat(_Inner(LlmTimeoutError("slow")))
    with pytest.raises(LlmTimeoutError):
        await _call(recorder)
    assert recorder.take() == [
        AttemptRecord(
            content=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            error="llm_timeout",
        )
    ]


async def test_recording_chat_does_not_record_other_exceptions() -> None:
    recorder = RecordingChat(_Inner(RuntimeError("bug")))
    with pytest.raises(RuntimeError):
        await _call(recorder)
    assert recorder.take() == []


OPTIONS = ClassifierOptions(
    max_input_chars=3000,
    max_output_tokens=256,
    temperature=0.0,
    timeout_s=120,
    disable_thinking=False,
)
NOT_JSON = ChatCompletion("not json", "stop", "m-test", 10, 5)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Slow:
    """A reply that first moves the fake clock by `seconds`."""

    def __init__(self, seconds: float, outcome: ChatCompletion | Exception) -> None:
        self.seconds = seconds
        self.outcome = outcome


Reply = ChatCompletion | Exception | Slow


class ScriptedChat:
    """ChatClient fake: replies per input text, every call moves the clock by `step_s`."""

    def __init__(self, clock: Clock, step_s: float = 1.0) -> None:
        self._clock = clock
        self._step_s = step_s
        self._script: dict[str, list[Reply]] = {}
        self.seen: list[str] = []

    def on(self, text: str, *replies: Reply) -> None:
        self._script[text] = list(replies)

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
        # Retries end with a correction message, so the input is the last INPUT_PREFIX one.
        last = next(
            m["content"] for m in reversed(messages) if m["content"].startswith(INPUT_PREFIX)
        )
        text = str(json.loads(last.removeprefix(INPUT_PREFIX)))
        self.seen.append(text)
        queue = self._script.get(text)
        reply: Reply = queue.pop(0) if queue else GOOD
        if isinstance(reply, Slow):
            self._clock.now += reply.seconds
            reply = reply.outcome
        else:
            self._clock.now += self._step_s
        if isinstance(reply, Exception):
            raise reply
        return reply


def _items(n: int, text: str = "error number {}") -> list[EvalItem]:
    return [
        EvalItem(
            f"t-{i:03d}",
            text.format(i),
            Category.RUNTIME_ERROR,
            Severity.MEDIUM,
            "python",
            (),
            None,
            "en",
            "note",
        )
        for i in range(1, n + 1)
    ]


async def test_all_items_ok() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, NOT_JSON, GOOD)
    out = await run_eval(items, chat, OPTIONS, max_s=1000, clock=clock)
    assert out.stop_reason is None
    assert [r.attempts for r in out.results] == [1, 2, 1]
    assert [r.outcome for r in out.results] == ["ok"] * 3
    log = out.results[1].attempt_log
    assert len(log) == 2
    assert log[0].content == "not json"
    assert log[1].content == GOOD.content


async def test_warmup_runs_first_and_is_not_recorded() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    out = await run_eval(_items(2), chat, OPTIONS, max_s=1000, clock=clock)
    assert chat.seen[0] == WARMUP_TEXT
    assert len(out.results) == 2
    assert all(len(r.attempt_log) == 1 for r in out.results)


async def test_warmup_model_error_raises() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    chat.on(WARMUP_TEXT, NOT_JSON, NOT_JSON)
    with pytest.raises(ModelOutputInvalidError):
        await run_eval(_items(3), chat, OPTIONS, max_s=1000, clock=clock)


async def test_warmup_unavailable_raises() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    chat.on(WARMUP_TEXT, LlmUnavailableError("down"))
    with pytest.raises(LlmUnavailableError):
        await run_eval(_items(2), chat, OPTIONS, max_s=1000, clock=clock)


async def test_warmup_bad_response_raises() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    chat.on(WARMUP_TEXT, LlmBadResponseError("odd"))
    with pytest.raises(LlmBadResponseError):
        await run_eval(_items(2), chat, OPTIONS, max_s=1000, clock=clock)


async def test_timeout_is_recorded_as_model_outcome() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, Slow(7, LlmTimeoutError("slow")))
    out = await run_eval(items, chat, OPTIONS, max_s=1000, clock=clock)
    row = out.results[1]
    assert (row.outcome, row.attempts, row.latency_ms) == ("llm_timeout", 1, 7000)
    assert row.category is None
    assert row.summary is None
    assert row.prompt_tokens is None
    assert [a.error for a in row.attempt_log] == ["llm_timeout"]
    assert len(out.results) == 3


async def test_time_limit_stops_before_next_item() -> None:
    clock = Clock()
    chat = ScriptedChat(clock, step_s=4)
    out = await run_eval(_items(5), chat, OPTIONS, max_s=10, clock=clock)
    assert len(out.results) == 2
    assert out.stop_reason == "time_limit"


async def test_limit_stops_after_n_items() -> None:
    clock = Clock()
    out = await run_eval(_items(3), ScriptedChat(clock), OPTIONS, max_s=1000, limit=2, clock=clock)
    assert len(out.results) == 2
    assert out.stop_reason == "limit"


async def test_limit_equal_to_size_runs_everything() -> None:
    clock = Clock()
    out = await run_eval(_items(3), ScriptedChat(clock), OPTIONS, max_s=1000, limit=3, clock=clock)
    assert len(out.results) == 3
    assert out.stop_reason is None


async def test_other_exception_propagates() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(2)
    chat.on(items[0].text, RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        await run_eval(items, chat, OPTIONS, max_s=1000, clock=clock)


async def test_on_item_is_called_per_result() -> None:
    clock = Clock()
    seen: list[ItemResult] = []
    out = await run_eval(
        _items(3), ScriptedChat(clock), OPTIONS, max_s=1000, clock=clock, on_item=seen.append
    )
    assert seen == out.results
    assert [r.id for r in seen] == ["t-001", "t-002", "t-003"]


async def test_truncated_flag_is_kept() -> None:
    clock = Clock()
    options = ClassifierOptions(
        max_input_chars=200,
        max_output_tokens=256,
        temperature=0.0,
        timeout_s=120,
        disable_thinking=False,
    )
    out = await run_eval(
        _items(1, "x" * 500), ScriptedChat(clock), options, max_s=1000, clock=clock
    )
    assert out.results[0].truncated is True


async def test_wall_time_is_measured_with_the_clock() -> None:
    clock = Clock()
    out = await run_eval(_items(2), ScriptedChat(clock, step_s=3), OPTIONS, max_s=1000, clock=clock)
    assert out.wall_s == 9.0


class Probe:
    """Readiness probe fake: fails `failures` times (forever when None), then passes."""

    def __init__(self, failures: int | None = 0) -> None:
        self.failures = failures
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1
        if self.failures is None or self.calls <= self.failures:
            raise LlmUnavailableError("still down")


class Pauses:
    def __init__(self) -> None:
        self.values: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.values.append(seconds)


async def test_unavailable_without_probe_stops() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, LlmUnavailableError("down"))
    out = await run_eval(items, chat, OPTIONS, max_s=1000, clock=clock)
    assert len(out.results) == 1
    assert out.stop_reason == "llm_unavailable"


async def test_unavailable_recovers_and_retries_the_row() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, LlmUnavailableError("down"), GOOD)
    pauses = Pauses()
    out = await run_eval(
        items, chat, OPTIONS, max_s=1000, probe=Probe(2), sleep=pauses, clock=clock
    )
    assert len(out.results) == 3
    assert out.stop_reason is None
    assert out.results[1].outcome == "ok"
    assert len(out.results[1].attempt_log) == 1
    assert pauses.values == [5.0, 5.0, 5.0]


async def test_unavailable_not_recovered_stops() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, LlmUnavailableError("down"))
    pauses = Pauses()
    probe = Probe(None)
    out = await run_eval(items, chat, OPTIONS, max_s=1000, probe=probe, sleep=pauses, clock=clock)
    assert len(out.results) == 1
    assert out.stop_reason == "llm_unavailable"
    assert len(pauses.values) == 12
    assert probe.calls == 12


async def test_unavailable_twice_on_the_row_stops() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, LlmUnavailableError("down"), LlmUnavailableError("down"))
    out = await run_eval(
        items, chat, OPTIONS, max_s=1000, probe=Probe(), sleep=Pauses(), clock=clock
    )
    assert len(out.results) == 1
    assert out.stop_reason == "llm_unavailable"


async def test_bad_response_on_the_retried_row_is_a_row_outcome() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, LlmUnavailableError("down"), LlmBadResponseError("odd"))
    out = await run_eval(
        items, chat, OPTIONS, max_s=1000, probe=Probe(), sleep=Pauses(), clock=clock
    )
    assert [r.outcome for r in out.results] == ["ok", "llm_bad_response", "ok"]
    assert len(out.results[1].attempt_log) == 1


async def test_one_bad_response_is_a_row_outcome() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, LlmBadResponseError("odd"))
    probe = Probe()
    seen: list[ItemResult] = []
    out = await run_eval(
        items, chat, OPTIONS, max_s=1000, probe=probe, clock=clock, on_item=seen.append
    )
    assert [r.outcome for r in out.results] == ["ok", "llm_bad_response", "ok"]
    assert out.stop_reason is None
    assert probe.calls == 1
    assert len(seen) == 3


async def test_bad_response_with_dead_runtime_stops() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(3)
    chat.on(items[1].text, LlmBadResponseError("odd"))
    out = await run_eval(items, chat, OPTIONS, max_s=1000, probe=Probe(None), clock=clock)
    assert [r.outcome for r in out.results] == ["ok", "llm_bad_response"]
    assert out.stop_reason == "llm_unavailable"


async def test_two_bad_responses_in_a_row_stop() -> None:
    clock = Clock()
    chat = ScriptedChat(clock)
    items = _items(4)
    chat.on(items[1].text, LlmBadResponseError("odd"))
    chat.on(items[2].text, LlmBadResponseError("odd"))
    probe = Probe()
    out = await run_eval(items, chat, OPTIONS, max_s=1000, probe=probe, clock=clock)
    assert len(out.results) == 3
    assert out.stop_reason == "llm_bad_response"
    assert probe.calls == 1
