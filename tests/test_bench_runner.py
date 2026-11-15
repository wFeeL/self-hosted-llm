import asyncio
import json
import os
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import httpx
import pytest
import respx

from error_classifier.benchmark.records import RequestRecord
from error_classifier.benchmark.runner import (
    BenchUnavailable,
    CommandResult,
    Plan,
    RunResult,
    Target,
    bench_inputs,
    cgroup_file,
    long_input,
    read_env,
    restart_count,
    run_bench,
    subprocess_runner,
)
from error_classifier.evaluation.dataset import DEFAULT_DATASET, load_dataset

pytestmark = pytest.mark.anyio

ROOT = Path(__file__).resolve().parents[1]
API = "http://api.test"
LLM = "http://llm.test"
FILES = ("compose.yaml", "compose.debug.yaml")
TARGET = Target(
    engine="podman",
    api_url=API,
    llm_url=LLM,
    llm_container="llm_c",
    api_container="api_c",
    compose_files=FILES,
)
COMPOSE = ["podman", "compose", "-f", "compose.yaml", "-f", "compose.debug.yaml"]
API_ENV = {
    "MAX_CONCURRENCY": "2",
    "MAX_QUEUE": "4",
    "QUEUE_TIMEOUT_S": "30",
    "REQUEST_TIMEOUT_S": "120",
    "LLM_MODEL": "m-test",
}
LLM_ENV = {"LLAMA_ARG_OFFLINE": "1", "LLAMA_ARG_THREADS": "6"}


def make_plan(
    *, cold_start: bool = False, skip_sequential: bool = False, max_s: float = 1e9
) -> Plan:
    return Plan(
        inputs=[(f"id{i}", f"text {i}") for i in range(20)],
        long_text="L" * 3000,
        cold_start=cold_start,
        skip_sequential=skip_sequential,
        max_s=max_s,
    )


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.pause = 0.0  # real seconds, lets the engine thread of a cold start finish

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(self.pause)


class Call:
    def __init__(self, args: Sequence[str], env: Mapping[str, str] | None, timeout_s: float):
        self.args = list(args)
        self.env = env
        self.timeout_s = timeout_s


class Stack:
    """Fake API, fake llama-server and fake engine commands sharing one clock."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.clock = Clock()
        self.calls: list[Call] = []
        self.latency_s = 0.1
        self.posts: list[str] = []
        self.timeouts: list[dict[str, float | None]] = []
        self.inflight = 0
        self.inflight_log: list[int] = []
        self.counters = {
            "prompt_tokens_total": 0.0,
            "prompt_tokens_cached_total": 0.0,
            "prompt_seconds_total": 0.0,
            "tokens_predicted_total": 0.0,
            "tokens_predicted_seconds_total": 0.0,
        }
        self.mode: Callable[[int], httpx.Response] = self._ok
        self.restarts = 0
        self.restart_at: int | None = None
        self.negative_at: int | None = None
        self.busy_polls = 0  # next N /metrics calls report a busy slot
        self.busy_always = False
        self.metrics_status = 200
        self.health_after = 0  # calls answered 503 before the first 200
        self.ready_after = 0
        self.ready_error: Exception | None = None
        self.health_calls = 0
        self.ready_calls = 0
        self.up_code = 0
        self.api_env: dict[str, str] | None = dict(API_ENV)
        self.llm_env: dict[str, str] | None = dict(LLM_ENV)
        self.cgroup_code = 0
        self.cpu = {"llm_c": 0, "api_c": 0}
        router.get(f"{API}/ready").mock(side_effect=self._ready)
        router.get(f"{LLM}/health").mock(side_effect=self._health)
        router.get(f"{LLM}/metrics").mock(side_effect=self._metrics)
        router.post(f"{API}/classify").mock(side_effect=self._classify)

    def _ok(self, n: int) -> httpx.Response:
        return httpx.Response(200, json={"category": "other", "model": "m-test"})

    def _ready(self, request: httpx.Request) -> httpx.Response:
        self.ready_calls += 1
        if self.ready_error is not None:
            raise self.ready_error
        status = 503 if self.ready_calls <= self.ready_after else 200
        return httpx.Response(status, json={"status": "ready"})

    def _health(self, request: httpx.Request) -> httpx.Response:
        self.health_calls += 1
        return httpx.Response(503 if self.health_calls <= self.health_after else 200)

    def _metrics(self, request: httpx.Request) -> httpx.Response:
        if self.metrics_status != 200:
            return httpx.Response(self.metrics_status)
        busy = self.busy_always or self.busy_polls > 0
        self.busy_polls = max(0, self.busy_polls - 1)
        lines = ["# HELP llamacpp:requests_processing Number of requests processing."]
        lines.append("# TYPE llamacpp:requests_processing gauge")
        lines.append(f"llamacpp:requests_processing {1 if busy else 0}")
        lines.extend(f"llamacpp:{k} {v}" for k, v in self.counters.items())
        return httpx.Response(200, text="\n".join(lines) + "\n")

    async def _classify(self, request: httpx.Request) -> httpx.Response:
        self.posts.append(json.loads(request.content)["message"])
        n = len(self.posts)
        self.timeouts.append(request.extensions["timeout"])
        self.inflight += 1
        self.inflight_log.append(self.inflight)
        await asyncio.sleep(0)
        self.inflight -= 1
        self.clock.now += self.latency_s
        self.counters["prompt_tokens_total"] += 100
        self.counters["prompt_tokens_cached_total"] += 20
        self.counters["prompt_seconds_total"] += 0.5
        self.counters["tokens_predicted_total"] += 10
        self.counters["tokens_predicted_seconds_total"] += 1.0
        if n == self.negative_at:
            self.counters["prompt_tokens_total"] -= 5000
        if n == self.restart_at:
            self.restarts += 1
        return self.mode(n)

    def runner(
        self, args: Sequence[str], *, env: Mapping[str, str] | None = None, timeout_s: float = 60.0
    ) -> CommandResult:
        self.calls.append(Call(args, env, timeout_s))
        engine_command = args[1]
        if engine_command == "compose":
            code = self.up_code if "up" in args else 0
            return CommandResult(code, "")
        container = args[-1]
        if engine_command == "inspect":
            template = args[3]
            if template == "{{.RestartCount}}":
                return CommandResult(0, f"{self.restarts}\n")
            env_map = self.api_env if container == "api_c" else self.llm_env
            if env_map is None:
                return CommandResult(1, "")
            return CommandResult(0, json.dumps([f"{k}={v}" for k, v in env_map.items()]))
        if engine_command == "exec":
            return self._cat(args[2], args[-1].rsplit("/", 1)[-1])
        return CommandResult(1, "")

    def _cat(self, container: str, name: str) -> CommandResult:
        if self.cgroup_code:
            return CommandResult(self.cgroup_code, "")
        if name == "cpu.stat":
            self.cpu[container] += 1_000_000
            return CommandResult(0, f"usage_usec {self.cpu[container]}\nuser_usec 5\n")
        if name == "memory.current":
            return CommandResult(0, "5000\n")
        return CommandResult(0, "anon 3000\nfile 700\n")

    async def run(
        self, plan: Plan | None = None, *, on_record: Callable[[RequestRecord], None] | None = None
    ) -> RunResult:
        async with httpx.AsyncClient() as http:
            return await run_bench(
                TARGET,
                plan or make_plan(),
                run=self.runner,
                http=http,
                clock=self.clock,
                sleep=self.clock.sleep,
                on_record=on_record,
            )


@pytest.fixture
def stack() -> Iterator[Stack]:
    with respx.mock(assert_all_called=False) as router:
        yield Stack(router)


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
    assert api.cpu_usage_usec == 1_000_000
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
    stack.latency_s = 1.0
    result = await stack.run(make_plan(max_s=10))

    assert result.stop_reason == "time_limit"
    assert not result.complete
    assert by_phase(result) == {"warmup": 3, "series": 20}


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
    assert subprocess_runner(["sleep", "5"], timeout_s=0.1) == CommandResult(-1, "")
    assert subprocess_runner(["/no/such/binary"]) == CommandResult(-1, "")
