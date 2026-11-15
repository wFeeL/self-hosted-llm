import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import httpx
import pytest
from bench_fakes import API_ENV, COMPOSE, LLM_ENV, Stack, make_plan

from error_classifier.benchmark.records import RequestRecord
from error_classifier.benchmark.runner import (
    BenchUnavailable,
    ColdStart,
    CommandResult,
    RunResult,
    bench_inputs,
    cgroup_file,
    long_input,
    read_env,
    restart_count,
    subprocess_runner,
)
from error_classifier.evaluation.dataset import DEFAULT_DATASET, load_dataset

pytestmark = pytest.mark.anyio

ROOT = Path(__file__).resolve().parents[1]


def by_phase(result: RunResult) -> Counter[str]:
    return Counter(r.phase for r in result.records)


async def test_full_run_without_cold_start(stack: Stack) -> None:
    seen: list[RequestRecord] = []
    result = await stack.run(on_record=seen.append)

    assert by_phase(result) == {
        "warmup": 3,
        "series": 20,
        "long": 1,
        "sequential": 50,
        "parallel": 20,
        "queue": 7,
    }
    assert result.complete
    assert result.stop_reason is None
    assert result.cold_start is None
    assert result.model == "m-test"
    assert result.api_env == API_ENV
    assert result.llm_env == LLM_ENV
    assert seen == result.records
    edges = Counter((r.phase, r.edge) for r in result.readings)
    assert edges == {
        (phase, edge): 1
        for phase in ("series", "long", "sequential", "parallel", "queue")
        for edge in ("start", "end")
    }
    # Timeout is QUEUE_TIMEOUT_S + REQUEST_TIMEOUT_S + 30, D1.
    assert {t["read"] for t in stack.timeouts} == {180.0}
    assert stack.posts[:3] == ["text 0", "text 1", "text 2"]


async def test_phases_are_announced_before_they_start(stack: Stack) -> None:
    names: list[str] = []
    await stack.run(make_plan(skip_sequential=True), on_phase=names.append)
    assert names == ["warmup", "series", "long", "parallel", "queue"]

    stopped: list[str] = []
    await stack.run(make_plan(max_s=0), on_phase=stopped.append)
    assert stopped == ["warmup"]


async def test_series_records_carry_tokens_and_timing(stack: Stack) -> None:
    result = await stack.run()
    series = [r for r in result.records if r.phase == "series"]

    assert [r.index for r in series] == list(range(20))
    assert {r.item_id for r in series} == {f"id{i}" for i in range(20)}
    for r in series:
        assert (r.status, r.outcome, r.wall_ms) == (200, "ok", 100)
        assert (r.prompt_tokens, r.cached_tokens, r.completion_tokens) == (120, 20, 10)
    assert series[0].started_s == pytest.approx(0.3)
    assert series[1].started_s == pytest.approx(0.4)
    untracked = [r for r in result.records if r.phase in ("warmup", "parallel", "queue")]
    assert all(r.prompt_tokens is None and r.completion_tokens is None for r in untracked)
    long = next(r for r in result.records if r.phase == "long")
    assert long.item_id == "long"
    assert long.prompt_tokens == 120


async def test_readings_hold_counters_and_cgroups(stack: Stack) -> None:
    result = await stack.run()
    start = next(r for r in result.readings if (r.phase, r.edge) == ("series", "start"))
    end = next(r for r in result.readings if (r.phase, r.edge) == ("series", "end"))

    assert start.metrics["llamacpp:tokens_predicted_total"] == 30
    assert end.metrics["llamacpp:tokens_predicted_total"] == 230
    assert start.at_s == pytest.approx(0.3)
    llm = next(c for c in start.cgroups if c.container == "llm")
    api = next(c for c in start.cgroups if c.container == "api")
    assert (llm.cpu_usage_usec, llm.memory_current, llm.memory_anon) == (1_000_000, 5000, 3000)
    assert (api.cpu_usage_usec, api.memory_current, api.memory_swap_current) == (1_000_000, 500, 4)
    assert llm.memory_swap_current == 40
    assert api.memory_anon == 300
    end_llm = next(c for c in end.cgroups if c.container == "llm")
    assert end_llm.cpu_usage_usec == 2_000_000


async def test_phase_inputs_and_concurrency(stack: Stack) -> None:
    result = await stack.run()

    sequential = [r for r in result.records if r.phase == "sequential"]
    assert [r.item_id for r in sequential] == [f"id{i % 20}" for i in range(50)]
    parallel = [r for r in result.records if r.phase == "parallel"]
    assert {r.index: r.client for r in parallel} == {i: i % 2 for i in range(20)}
    queue = [r for r in result.records if r.phase == "queue"]
    assert sorted(r.index for r in queue) == list(range(7))
    assert [r.item_id for r in sorted(queue, key=lambda r: r.index)] == [f"id{i}" for i in range(7)]
    log = stack.inflight_log
    assert max(log[:74]) == 1  # warmup, series, long, sequential go one by one
    assert max(log[74:94]) == 2  # two clients
    assert max(log[94:]) == 7  # the queue probe sends all at once


async def test_skip_sequential(stack: Stack) -> None:
    result = await stack.run(make_plan(skip_sequential=True))

    assert "sequential" not in by_phase(result)
    assert all(r.phase != "sequential" for r in result.readings)
    assert result.complete
    assert len(stack.posts) == 3 + 20 + 1 + 20 + 7


async def test_cold_start_runs_down_then_up_offline(stack: Stack) -> None:
    stack.health_after = 2
    stack.ready_after = 4
    result = await stack.run(make_plan(cold_start=True))

    compose = [c for c in stack.calls if c.args[1] == "compose"]
    assert [c.args for c in compose] == [[*COMPOSE, "down"], [*COMPOSE, "up", "-d"]]
    up = compose[1]
    assert up.env is not None
    assert up.env["LLAMA_ARG_OFFLINE"] == "1"
    assert up.env["PATH"] == os.environ["PATH"]
    assert up.timeout_s == 700
    assert result.cold_start is not None
    assert result.cold_start.health_s == pytest.approx(1.0)
    assert result.cold_start.ready_s == pytest.approx(2.0)
    assert result.complete


async def test_cold_start_that_never_gets_ready(stack: Stack) -> None:
    stack.ready_after = 10**9
    result = await stack.run(make_plan(cold_start=True))

    assert result.stop_reason == "cold_start_failed"
    assert not result.complete
    assert result.records == []
    assert result.cold_start is not None
    assert result.cold_start.health_s == 0
    assert result.cold_start.ready_s is None
    assert stack.clock.now - 1000.0 >= 600


async def test_cold_start_stops_when_up_fails(stack: Stack) -> None:
    stack.ready_after = 10**9
    stack.up_code = 1
    stack.clock.pause = 0.002
    result = await stack.run(make_plan(cold_start=True))

    assert result.stop_reason == "cold_start_failed"
    assert stack.clock.now - 1000.0 < 600


async def test_cold_start_stops_when_down_fails(stack: Stack) -> None:
    stack.down_code = 1
    stack.down_stderr = "no such pod"
    result = await stack.run(make_plan(cold_start=True))

    assert result.stop_reason == "cold_start_failed"
    assert result.failure_detail == "no such pod"
    assert result.cold_start == ColdStart(health_s=None, ready_s=None)
    assert all("up" not in c.args for c in stack.calls)


async def test_cold_start_stops_when_the_old_stack_still_answers(stack: Stack) -> None:
    stack.alive_after_down = True
    result = await stack.run(make_plan(cold_start=True))

    assert result.stop_reason == "cold_start_failed"
    assert "still answers" in result.failure_detail
    assert result.cold_start == ColdStart(health_s=None, ready_s=None)
    assert all("up" not in c.args for c in stack.calls)


async def test_failed_up_reports_the_error_tail(stack: Stack) -> None:
    stack.ready_after = 10**9
    stack.up_code = 1
    stack.up_stderr = "port is busy"
    stack.clock.pause = 0.002
    result = await stack.run(make_plan(cold_start=True))

    assert result.failure_detail == "port is busy"


async def test_cold_start_stops_without_offline(stack: Stack) -> None:
    stack.llm_env = {"LLAMA_ARG_OFFLINE": "0"}
    result = await stack.run(make_plan(cold_start=True))

    assert result.stop_reason == "not_offline"
    assert result.records == []
    assert result.cold_start is not None
    assert result.cold_start.ready_s == 0


@pytest.mark.parametrize("error", [None, httpx.ConnectError("refused")])
async def test_api_not_ready(stack: Stack, error: Exception | None) -> None:
    stack.ready_after = 10**9
    stack.ready_error = error
    with pytest.raises(BenchUnavailable):
        await stack.run()
    assert stack.posts == []


@pytest.mark.parametrize(
    "env",
    [None, {"MAX_QUEUE": "4"}, {**API_ENV, "QUEUE_TIMEOUT_S": "soon"}],
)
async def test_unusable_api_env(stack: Stack, env: dict[str, str] | None) -> None:
    stack.api_env = env
    with pytest.raises(BenchUnavailable):
        await stack.run()


async def test_queue_size_follows_api_env(stack: Stack) -> None:
    stack.api_env = {**API_ENV, "MAX_CONCURRENCY": "1", "MAX_QUEUE": "1"}
    result = await stack.run()

    assert by_phase(result)["queue"] == 3
    assert {t["read"] for t in stack.timeouts} == {180.0}


def reply(response: httpx.Response) -> Callable[[int], httpx.Response]:
    def mode(n: int) -> httpx.Response:
        return response

    return mode


@pytest.mark.parametrize(
    ("response", "outcome", "status"),
    [
        (
            httpx.Response(429, json={"error": {"code": "queue_full", "message": "x"}}),
            "queue_full",
            429,
        ),
        (httpx.Response(502, text="bad gateway"), "http_502", 502),
        (httpx.Response(500, json={"detail": "x"}), "http_500", 500),
        (httpx.Response(500, json={"error": {"code": 7}}), "http_500", 500),
    ],
)
async def test_http_outcomes(
    stack: Stack, response: httpx.Response, outcome: str, status: int
) -> None:
    stack.mode = reply(response)
    result = await stack.run()

    assert {r.outcome for r in result.records} == {outcome}
    assert {r.status for r in result.records} == {status}
    assert result.model is None
    assert result.complete


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (httpx.ReadTimeout("slow"), "client_timeout"),
        (httpx.ConnectError("refused"), "transport_error"),
    ],
)
async def test_transport_outcomes(stack: Stack, error: Exception, outcome: str) -> None:
    def fail(n: int) -> httpx.Response:
        raise error

    stack.mode = fail
    result = await stack.run()

    assert {r.outcome for r in result.records} == {outcome}
    assert {r.status for r in result.records} == {None}


async def test_busy_runtime_is_recorded_without_a_request(stack: Stack) -> None:
    stack.busy_always = True
    result = await stack.run()

    measured = [r for r in result.records if r.phase in ("series", "long", "sequential")]
    assert len(measured) == 71
    assert {(r.outcome, r.status, r.wall_ms) for r in measured} == {("runtime_busy", None, 30000)}
    assert all(r.prompt_tokens is None for r in measured)
    assert len(stack.posts) == 3 + 20 + 7  # warmup, parallel, queue do not wait


async def test_waits_until_the_runtime_is_idle(stack: Stack) -> None:
    stack.busy_polls = 3  # series start reading, then two polls
    result = await stack.run()

    assert stack.clock.sleeps.count(0.25) == 2
    first = next(r for r in result.records if r.phase == "series")
    assert first.outcome == "ok"


async def test_restart_stops_the_run(stack: Stack) -> None:
    stack.restart_at = 5
    result = await stack.run()

    assert result.stop_reason == "runtime_restarted"
    assert not result.complete
    assert by_phase(result) == {"warmup": 3, "series": 20}
    assert (("series", "end")) in {(r.phase, r.edge) for r in result.readings}


async def test_negative_counter_stops_the_run(stack: Stack) -> None:
    stack.negative_at = 10
    result = await stack.run()

    assert result.stop_reason == "negative_counter"
    assert not result.complete
    assert by_phase(result) == {"warmup": 3, "series": 20}


async def test_time_limit_stops_before_the_next_phase(stack: Stack) -> None:
    stack.latency_s = 0.45
    result = await stack.run(make_plan(max_s=10.3))

    assert result.stop_reason == "time_limit"
    assert not result.complete
    assert by_phase(result) == {"warmup": 3, "series": 20}


async def test_time_limit_stops_inside_the_series(stack: Stack) -> None:
    stack.latency_s = 1.0
    result = await stack.run(make_plan(max_s=5))

    assert result.stop_reason == "time_limit"
    assert not result.complete
    assert by_phase(result) == {"warmup": 3, "series": 2}
    assert [(r.phase, r.edge) for r in result.readings] == [("series", "start"), ("series", "end")]


async def test_time_limit_stops_inside_the_sequential_phase(stack: Stack) -> None:
    result = await stack.run(make_plan(max_s=4.5))

    assert result.stop_reason == "time_limit"
    counts = by_phase(result)
    assert counts["series"] == 20
    assert 0 < counts["sequential"] < 50
    assert "parallel" not in counts
    assert ("sequential", "end") in {(r.phase, r.edge) for r in result.readings}


async def test_time_limit_stops_both_clients_of_the_parallel_phase(stack: Stack) -> None:
    result = await stack.run(make_plan(skip_sequential=True, max_s=2.7))

    assert result.stop_reason == "time_limit"
    counts = by_phase(result)
    assert 0 < counts["parallel"] < 20
    assert "queue" not in counts
    assert ("parallel", "end") in {(r.phase, r.edge) for r in result.readings}


async def test_time_limit_does_not_cut_the_warmup(stack: Stack) -> None:
    result = await stack.run(make_plan(max_s=0))

    assert result.stop_reason == "time_limit"
    assert by_phase(result) == {"warmup": 3}
    assert result.readings == []


async def test_missing_metrics_endpoint(stack: Stack) -> None:
    stack.metrics_status = 404
    with pytest.raises(BenchUnavailable, match="/metrics"):
        await stack.run()


async def test_unreadable_cgroups_leave_empty_fields(stack: Stack) -> None:
    stack.cgroup_code = 1
    result = await stack.run()

    assert result.complete
    cgroup = result.readings[0].cgroups[0]
    assert (cgroup.cpu_usage_usec, cgroup.memory_current, cgroup.memory_anon) == (None, None, None)
    assert cgroup.memory_swap_current is None


def test_bench_inputs_take_every_third_item() -> None:
    items = load_dataset(ROOT / DEFAULT_DATASET)
    rows = bench_inputs(items)

    assert len(rows) == 20
    assert [i for i, _ in rows] == [items[k].id for k in range(0, 58, 3)]
    assert rows[1] == (items[3].id, items[3].text)


def test_bench_inputs_need_enough_items() -> None:
    with pytest.raises(ValueError, match="58"):
        bench_inputs(load_dataset(ROOT / DEFAULT_DATASET)[:40])


def test_long_input_is_stable_base64() -> None:
    text = long_input()

    assert text == long_input()
    assert len(text) == 3000
    assert text.isascii()
    assert set(text) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")


def test_cgroup_file_and_helpers() -> None:
    seen: list[list[str]] = []

    def run(args: Sequence[str], *, env: Mapping[str, str] | None = None, timeout_s: float = 60.0):
        seen.append(list(args))
        if args[1] == "inspect" and args[3] == "{{json .Config.Env}}":
            return CommandResult(0, '["A=1", "B=x=y", "NOEQ"]')
        if args[1] == "inspect":
            return CommandResult(0, "3\n")
        return CommandResult(0 if args[-1].endswith("memory.peak") else 1, "42\n")

    assert cgroup_file(run, "podman", "c", "memory.peak") == "42\n"
    assert seen[0] == ["podman", "exec", "c", "cat", "/sys/fs/cgroup/memory.peak"]
    assert cgroup_file(run, "podman", "c", "cpu.stat") is None
    assert read_env(run, "podman", "c") == {"A": "1", "B": "x=y"}
    assert restart_count(run, "podman", "c") == 3
    assert seen[-1] == ["podman", "inspect", "--format", "{{.RestartCount}}", "c"]


def test_helpers_on_failure() -> None:
    def run(args: Sequence[str], *, env: Mapping[str, str] | None = None, timeout_s: float = 60.0):
        return CommandResult(1, "")

    def garbage(
        args: Sequence[str], *, env: Mapping[str, str] | None = None, timeout_s: float = 60.0
    ):
        return CommandResult(0, "not json")

    assert read_env(run, "podman", "c") is None
    assert read_env(garbage, "podman", "c") is None
    assert restart_count(run, "podman", "c") is None
    assert restart_count(garbage, "podman", "c") is None


def test_subprocess_runner() -> None:
    assert subprocess_runner(["echo", "hi"]) == CommandResult(0, "hi\n")
    assert subprocess_runner(["sh", "-c", "echo $BENCH_X"], env={"BENCH_X": "1"}).stdout == "1\n"
    assert subprocess_runner(["sh", "-c", "exit 3"]).code == 3
    failed = subprocess_runner(
        ["sh", "-c", "echo a >&2; head -c 600 /dev/zero | tr '\\0' x >&2; exit 1"]
    )
    assert failed.code == 1
    assert len(failed.stderr_tail) == 500
    assert failed.stderr_tail.endswith("x")
    assert subprocess_runner(["sleep", "5"], timeout_s=0.1) == CommandResult(-1, "")
    assert subprocess_runner(["/no/such/binary"]) == CommandResult(-1, "")
