import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from error_classifier.classify.prompt import INPUT_PREFIX
from error_classifier.classify.schema import Category, Severity
from error_classifier.classify.service import ClassifierOptions
from error_classifier.errors import LlmTimeoutError, LlmUnavailableError
from error_classifier.evaluation.dataset import EvalItem, load_dataset
from error_classifier.llm.client import ChatCompletion, ChatMessage
from error_classifier.selection.check import (
    CandidateResult,
    CheckLimits,
    check_candidate,
    load_failed_result,
    skipped_result,
    smoke_items,
)
from error_classifier.selection.presets import Candidate
from error_classifier.selection.router import LoadOutcome, ModelStatus

pytestmark = pytest.mark.anyio

DATASET = Path(__file__).parent.parent / "eval" / "dataset.jsonl"
ALIAS = "m"
SECRET = "sk-secret-12345678"
LONG_TEXT = "long input marker"
CANDIDATE = Candidate(ALIAS, "org/name", "m.gguf", False)
OPTIONS = ClassifierOptions(3000, 256, 0.0, 120.0, False)
LOADED = LoadOutcome(True, 12.0, None, None)


def _item(i: int, secret: str | None = None) -> EvalItem:
    return EvalItem(
        f"t-{i}", f"error number {i}", Category.RUNTIME_ERROR, Severity.MEDIUM,
        "python", (), secret, "en", "note",
    )  # fmt: skip


SMOKE = [_item(1), _item(2, SECRET), _item(3), _item(4)]
ROW_TEXTS = [item.text for item in SMOKE]


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeChat:
    """ChatClient fake: answers by input text, every call moves the clock."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.model = ALIAS
        self.step_s = 1.0
        self.calls = 0
        self.timeouts: dict[str, float] = {}
        self.leaks: set[str] = set()
        self.invalid: set[str] = set()
        self.errors: dict[int, Exception] = {}  # call number -> raise
        self.steps: dict[int, float] = {}  # call number -> seconds on the clock
        self.on_call: Callable[[int], None] | None = None

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
        self.calls += 1
        number = self.calls
        last = next(
            m["content"] for m in reversed(messages) if m["content"].startswith(INPUT_PREFIX)
        )
        text = str(json.loads(last.removeprefix(INPUT_PREFIX)))
        self.timeouts[text] = timeout_s
        self.clock.now += self.steps.get(number, self.step_s)
        if self.on_call is not None:
            self.on_call(number)
        if number in self.errors:
            raise self.errors[number]
        if text in self.invalid:
            return ChatCompletion("not json", "stop", self.model, 10, 5)
        summary = f"The service failed with {SECRET}." if text in self.leaks else "It failed."
        answer = {
            "category": "runtime_error",
            "severity": "medium",
            "summary": summary,
            "suggested_action": "Check the code.",
        }
        return ChatCompletion(json.dumps(answer), "stop", self.model, 10, 5)


class FakeRouter:
    def __init__(self, load: LoadOutcome = LOADED) -> None:
        self.outcome = load
        self.loads: list[str] = []
        self.status = ModelStatus("loaded", False, None)

    async def load(self, alias: str, timeout_s: float) -> LoadOutcome:
        self.loads.append(alias)
        return self.outcome

    async def statuses(self) -> dict[str, ModelStatus]:
        return {ALIAS: self.status}


def _write_status(proc: Path, pid: int, rss_kb: int, hwm_kb: int) -> None:
    (proc / str(pid) / "status").write_text(
        f"Name:\tllama\nVmHWM:\t{hwm_kb} kB\nRssAnon:\t{rss_kb} kB\n", encoding="utf-8"
    )


@dataclass
class Env:
    clock: Clock
    chat: FakeChat
    router: FakeRouter
    proc: Path
    models: Path
    logs: list[str]

    async def run(
        self,
        *,
        model_max_s: float = 1000.0,
        target_p95_s: float = 30.0,
        smoke: Sequence[EvalItem] = SMOKE,
    ) -> CandidateResult:
        return await check_candidate(
            CANDIDATE,
            router=self.router,
            chat=self.chat,
            options=OPTIONS,
            smoke=smoke,
            long_text=LONG_TEXT,
            limits=CheckLimits(60.0, model_max_s, target_p95_s),
            proc=self.proc,
            models_dir=self.models,
            clock=self.clock,
            log=self.logs.append,
        )


def _env(tmp_path: Path, *, child: bool = True, child_file: str = "m.gguf") -> Env:
    proc = tmp_path / "proc"
    proc.mkdir()
    if child:
        (proc / "100").mkdir()
        argv = ("llama-server", "--alias", ALIAS, "--hf-file", child_file)
        (proc / "100" / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
        _write_status(proc, 100, 1000, 2000)
    models = tmp_path / "models"
    blob = models / "models--org--name" / "snapshots" / "x" / "m.gguf"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 5000)
    clock = Clock()
    return Env(clock, FakeChat(clock), FakeRouter(), proc, models, [])


def test_smoke_items_covers_secrets_and_categories() -> None:
    items = load_dataset(DATASET)
    secret_rows = [item for item in items if item.secret is not None]
    chosen = smoke_items(items)
    assert len(secret_rows) == 7
    assert {item.category for item in chosen} == set(Category)
    assert [item.id for item in chosen] == [i.id for i in items if i in chosen]
    assert all(row in chosen for row in secret_rows)
    assert len(chosen) == 12


def test_smoke_items_adds_first_row_of_a_missing_category() -> None:
    rows = [
        EvalItem("a", "t", Category.UNKNOWN, Severity.LOW, "go", (), None, "en", ""),
        EvalItem("b", "t", Category.NETWORK_ERROR, Severity.LOW, "go", (), "12345678", "en", ""),
        EvalItem("c", "t", Category.UNKNOWN, Severity.LOW, "go", (), None, "en", ""),
    ]
    assert [item.id for item in smoke_items(rows)] == ["a", "b"]


async def test_check_passes(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = await env.run()
    assert result.failures == []
    assert result.passed
    assert result.ok == 4
    assert result.smoke_n == 4
    assert result.p50_ms == 1000
    assert result.p95_ms == 1000
    assert result.long_outcome == "ok"
    assert result.long_ms == 1000
    assert result.hf_file_seen == "m.gguf"
    assert result.restarted is False
    assert result.warnings == []
    assert env.router.loads == [ALIAS]
    assert env.chat.calls == 1 + 4 + 1  # warm-up, rows, long input


async def test_check_logs_progress_without_error_text(tmp_path: Path) -> None:
    env = _env(tmp_path)
    await env.run()
    assert env.logs[0] == "m: loaded in 12 s"
    assert "m: row 1/4 ok 1.0 s" in env.logs
    assert env.logs[-1] == "m: passed"
    assert not any("error number" in line or LONG_TEXT in line for line in env.logs)


async def test_check_load_failure_stops(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.router.outcome = LoadOutcome(False, 5.0, "failed to load (exit code 137)", 137)
    result = await env.run()
    assert result.failures == ["load: failed to load (exit code 137)"]
    assert not result.passed
    assert env.chat.calls == 0


async def test_check_warmup_error_fails(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.errors = {1: LlmTimeoutError("slow")}
    result = await env.run()
    assert result.failures == ["warm-up failed: llm_timeout"]
    assert result.results == []
    assert result.long_outcome is None
    assert result.need_bytes is None
    assert result.file_size is None
    assert env.chat.calls == 1


async def test_check_warmup_runtime_stop_fails(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.errors = {1: LlmUnavailableError("down")}
    result = await env.run()
    assert result.failures == ["runtime stopped during the warm-up"]
    assert env.chat.calls == 1


async def test_check_runtime_stop_fails(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.errors = {4: LlmUnavailableError("down")}  # the third row
    result = await env.run()
    assert "smoke stopped after 2 of 4 rows (llm_unavailable)" in result.failures
    assert result.stop_reason == "llm_unavailable"
    assert not result.passed
    assert env.chat.calls == 4  # no retry of the row, no long input
    assert result.long_outcome is None
    assert not any(f.startswith("long input") for f in result.failures)


async def test_check_budget_stops_smoke(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = await env.run(model_max_s=3.0)
    assert "smoke stopped after 2 of 4 rows (time_limit)" in result.failures
    assert not result.passed


async def test_check_no_time_for_long_input(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = await env.run(model_max_s=8.0)
    assert result.failures == ["no time left for the long input"]
    assert result.long_outcome is None
    assert env.chat.calls == 1 + 4


async def test_check_two_invalid_rows_fail(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.invalid = {ROW_TEXTS[0], ROW_TEXTS[2]}
    result = await env.run()
    assert result.ok == 2
    assert "valid answers 2 of 4" in result.failures
    assert not any(f.startswith("smoke stopped") for f in result.failures)


async def test_check_one_invalid_row_is_tolerated(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.invalid = {ROW_TEXTS[0]}
    result = await env.run()
    assert result.ok == 3
    assert result.failures == []


async def test_check_leak_fails(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.leaks = {ROW_TEXTS[1]}
    result = await env.run()
    assert result.leaks == 1
    assert result.failures == ["leaked 1 secret(s)"]


async def test_check_other_model_fails(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.model = "other"
    result = await env.run()
    assert result.mismatches == 1 + 4 + 1  # warm-up, rows and the long input
    assert result.failures == ["answers came from another model"]


async def test_check_other_file_fails(tmp_path: Path) -> None:
    env = _env(tmp_path, child_file="other.gguf")
    result = await env.run()
    assert result.hf_file_seen == "other.gguf"
    assert result.failures == ["the runtime loaded another file"]


async def test_check_restart_by_pid(tmp_path: Path) -> None:
    env = _env(tmp_path)

    def restart(number: int) -> None:
        if number == 3:
            (env.proc / "100").rename(env.proc / "101")

    env.chat.on_call = restart
    result = await env.run()
    assert result.restarted
    assert result.failures == ["the runtime restarted (likely out of memory)"]


async def test_check_vanished_process_is_a_restart(tmp_path: Path) -> None:
    env = _env(tmp_path)

    def vanish(number: int) -> None:
        if number == 3:
            (env.proc / "100" / "cmdline").unlink()

    env.chat.on_call = vanish
    result = await env.run()
    assert result.restarted


async def test_check_restart_by_status(tmp_path: Path) -> None:
    env = _env(tmp_path, child=False)
    env.router.status = ModelStatus("unloaded", False, None)
    result = await env.run()
    assert result.restarted
    assert result.failures == ["the runtime restarted (likely out of memory)"]
    assert result.rss_anon_max is None
    assert result.need_bytes is None


async def test_check_invisible_process_and_loaded_status_is_fine(tmp_path: Path) -> None:
    env = _env(tmp_path, child=False)
    result = await env.run()
    assert not result.restarted
    assert result.failures == []
    assert result.hf_file_seen is None


async def test_check_slow_fails(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.step_s = 40.0
    result = await env.run()
    assert result.p95_ms == 40000
    assert result.failures == ["p95 40.0 s above 30 s"]


async def test_check_p95_counts_errors(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.errors = {3: LlmTimeoutError("slow")}
    env.chat.steps = {3: 130.0}
    result = await env.run(model_max_s=1200.0)
    assert result.ok == 3
    assert result.p50_ms == 1000
    assert result.p95_ms == 130000
    assert result.failures == ["p95 130.0 s above 30 s"]


async def test_check_p95_unknown_without_rows(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = await env.run(model_max_s=1.0)
    assert result.results == []
    assert "p95 unknown" in result.failures


async def test_check_long_input_gets_remaining_budget(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = await env.run(model_max_s=1000.0)
    # the warm-up and four rows took 5 s of the budget
    assert env.chat.timeouts[LONG_TEXT] == 995.0
    assert env.chat.timeouts[ROW_TEXTS[0]] == OPTIONS.timeout_s
    assert result.failures == []


async def test_check_long_input_failure_fails(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.chat.errors = {6: LlmTimeoutError("slow")}
    result = await env.run()
    assert result.long_outcome == "llm_timeout"
    assert result.failures == ["long input: llm_timeout"]


async def test_check_need_bytes(tmp_path: Path) -> None:
    env = _env(tmp_path)

    def grow(number: int) -> None:
        if number == 6:  # the long input
            _write_status(env.proc, 100, 3000, 4000)
        if number == 3:
            _write_status(env.proc, 100, 5000, 6000)

    env.chat.on_call = grow
    result = await env.run()
    assert result.rss_anon_max == 5000 * 1024
    assert result.vm_hwm == 4000 * 1024  # the last reading
    assert result.file_size == 5000
    assert result.need_bytes == 5000 * 1024 + 5000


async def test_check_missing_model_file_warns(tmp_path: Path) -> None:
    env = _env(tmp_path)
    (env.models / "models--org--name" / "snapshots" / "x" / "m.gguf").unlink()
    result = await env.run()
    assert result.warnings == ["model file not found in the models volume"]
    assert result.need_bytes is None
    assert result.passed


def test_factories() -> None:
    skipped = skipped_result("a", "too big")
    assert skipped.skipped == "too big"
    assert not skipped.passed
    assert skipped.failures == []
    failed = load_failed_result("a", LoadOutcome(False, 1.0, "boom", None))
    assert failed.failures == ["load: boom"]
    assert not failed.passed
