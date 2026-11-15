import asyncio
import base64
import json
import os
import random
import subprocess
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast

import httpx
from pydantic import BaseModel, ConfigDict

from error_classifier.benchmark.records import (
    CgroupReading,
    Phase,
    Reading,
    RequestRecord,
    parse_cpu_stat,
    parse_int,
    parse_memory_stat,
    parse_metrics,
)
from error_classifier.evaluation.dataset import EvalItem

PROBE_TIMEOUT_S: Final = 2.0
COLD_START_MAX_S: Final = 600.0
COLD_POLL_S: Final = 0.5
IDLE_POLL_S: Final = 0.25
IDLE_MAX_S: Final = 30.0
SLOW_COMMAND_TIMEOUT_S: Final = 700.0  # compose up waits for the llm healthcheck, build is slow
STDERR_TAIL: Final = 500
WARMUP_ROWS: Final = 3
SEQUENTIAL_ROWS: Final = 50
MIN_ITEMS: Final = 58
LONG_CHARS: Final = 3000
CLIENTS: Final = 2
EXTRA_TIMEOUT_S: Final = 30.0

_MEASURED: Final[frozenset[Phase]] = frozenset({"series", "long", "sequential"})
_TIMED: Final[frozenset[Phase]] = frozenset({"series", "sequential"})
_PROMPT: Final = "llamacpp:prompt_tokens_total"
_CACHED: Final = "llamacpp:prompt_tokens_cached_total"
_PREDICTED: Final = "llamacpp:tokens_predicted_total"
_PROCESSING: Final = "llamacpp:requests_processing"


class BenchUnavailable(Exception):
    pass


@dataclass(frozen=True)
class CommandResult:
    code: int
    stdout: str
    stderr_tail: str = ""


class Runner(Protocol):
    def __call__(
        self, args: Sequence[str], *, env: Mapping[str, str] | None = None, timeout_s: float = 60.0
    ) -> CommandResult: ...


@dataclass(frozen=True)
class Target:
    engine: Literal["podman", "docker"]
    api_url: str  # http://127.0.0.1:8000
    llm_url: str  # http://127.0.0.1:8080 (root, no /v1)
    llm_container: str
    api_container: str
    compose_files: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    inputs: list[tuple[str, str]]  # (item_id, text), 20 rows
    long_text: str
    cold_start: bool
    skip_sequential: bool
    max_s: float


class ColdStart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    health_s: float | None
    ready_s: float | None


@dataclass(frozen=True)
class RunResult:
    records: list[RequestRecord]
    readings: list[Reading]
    cold_start: ColdStart | None
    complete: bool
    # "time_limit", "runtime_restarted", "negative_counter", "cold_start_failed", "not_offline"
    stop_reason: str | None
    model: str | None  # "model" of the first ok answer
    api_env: dict[str, str]
    llm_env: dict[str, str]
    failure_detail: str = ""  # stderr tail of the command that failed a cold start


def subprocess_runner(
    args: Sequence[str], *, env: Mapping[str, str] | None = None, timeout_s: float = 60.0
) -> CommandResult:
    try:
        proc = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=None if env is None else dict(env),
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return CommandResult(-1, "")
    return CommandResult(proc.returncode, proc.stdout, proc.stderr[-STDERR_TAIL:])


def bench_inputs(items: Sequence[EvalItem]) -> list[tuple[str, str]]:
    if len(items) < MIN_ITEMS:
        raise ValueError(f"the benchmark needs at least {MIN_ITEMS} items, got {len(items)}")
    return [(items[i].id, items[i].text) for i in range(0, MIN_ITEMS, 3)]


def long_input() -> str:
    return base64.b64encode(random.Random(7).randbytes(2400)).decode()[:LONG_CHARS]


def cgroup_file(run: Runner, engine: str, container: str, name: str) -> str | None:
    result = run([engine, "exec", container, "cat", f"/sys/fs/cgroup/{name}"])
    return result.stdout if result.code == 0 else None


def read_env(run: Runner, engine: str, container: str) -> dict[str, str] | None:
    result = run([engine, "inspect", "--format", "{{json .Config.Env}}", container])
    if result.code != 0:
        return None
    try:
        data = cast(object, json.loads(result.stdout))
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    env: dict[str, str] = {}
    for entry in cast("list[object]", data):
        if not isinstance(entry, str):
            return None
        key, sep, value = entry.partition("=")
        if sep:
            env[key] = value
    return env


def restart_count(run: Runner, engine: str, container: str) -> int | None:
    result = run([engine, "inspect", "--format", "{{.RestartCount}}", container])
    return parse_int(result.stdout) if result.code == 0 else None


def _error_code(response: httpx.Response) -> str | None:
    try:
        body = cast(object, response.json())
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    error = cast("dict[str, object]", body).get("error")
    if not isinstance(error, dict):
        return None
    code = cast("dict[str, object]", error).get("code")
    return code if isinstance(code, str) else None


def _answer_model(response: httpx.Response) -> str | None:
    try:
        body = cast(object, response.json())
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    model = cast("dict[str, object]", body).get("model")
    return model if isinstance(model, str) else None


def _seconds_setting(env: Mapping[str, str], key: str) -> float:
    try:
        return float(env[key])
    except KeyError, ValueError:
        raise BenchUnavailable(f"API env {key} is missing or not a number") from None


def _count_setting(env: Mapping[str, str], key: str) -> int:
    try:
        return int(env[key])
    except KeyError, ValueError:
        raise BenchUnavailable(f"API env {key} is missing or not an integer") from None


class _Bench:
    def __init__(
        self,
        target: Target,
        run: Runner,
        http: httpx.AsyncClient,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
        on_record: Callable[[RequestRecord], None] | None,
        max_s: float,
        api_key: str | None = None,
    ) -> None:
        self.api_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.target = target
        self.run = run
        self.http = http
        self.clock = clock
        self.sleep = sleep
        self.on_record = on_record
        self.start = clock()
        self.client_timeout_s = 0.0
        self.records: list[RequestRecord] = []
        self.readings: list[Reading] = []
        self.model: str | None = None
        self.negative = False
        self.max_s = max_s
        self.out_of_time = False

    async def get_ok(self, url: str) -> bool:
        try:
            # The key goes to the API only, never to llama-server.
            headers = self.api_headers if url.startswith(self.target.api_url) else {}
            response = await self.http.get(url, headers=headers, timeout=PROBE_TIMEOUT_S)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def metrics(self) -> dict[str, float]:
        try:
            response = await self.http.get(
                f"{self.target.llm_url}/metrics", timeout=PROBE_TIMEOUT_S
            )
        except httpx.HTTPError:
            return {}
        return parse_metrics(response.text) if response.status_code == 200 else {}

    def cgroup(self, container: Literal["llm", "api"]) -> CgroupReading:
        name = self.target.llm_container if container == "llm" else self.target.api_container
        engine = self.target.engine
        cpu = cgroup_file(self.run, engine, name, "cpu.stat")
        current = cgroup_file(self.run, engine, name, "memory.current")
        stat = cgroup_file(self.run, engine, name, "memory.stat")
        return CgroupReading(
            container=container,
            cpu_usage_usec=None if cpu is None else parse_cpu_stat(cpu),
            memory_current=None if current is None else parse_int(current),
            memory_anon=None if stat is None else parse_memory_stat(stat).get("anon"),
        )

    async def take_reading(self, phase: Phase, edge: Literal["start", "end"]) -> None:
        at_s = self.clock() - self.start
        self.readings.append(
            Reading(
                phase=phase,
                edge=edge,
                at_s=at_s,
                metrics=await self.metrics(),
                cgroups=[self.cgroup("llm"), self.cgroup("api")],
            )
        )

    def time_is_up(self) -> bool:
        if self.clock() - self.start >= self.max_s:
            self.out_of_time = True
        return self.out_of_time

    def add(self, record: RequestRecord) -> None:
        self.records.append(record)
        if self.on_record is not None:
            self.on_record(record)

    async def wait_idle(self) -> tuple[dict[str, float], bool]:
        began = self.clock()
        while True:
            snapshot = await self.metrics()
            if snapshot.get(_PROCESSING, 0.0) == 0:
                return snapshot, False
            if self.clock() - began >= IDLE_MAX_S:
                return snapshot, True
            await self.sleep(IDLE_POLL_S)

    def tokens(
        self, before: Mapping[str, float], after: Mapping[str, float]
    ) -> tuple[int | None, int | None, int | None]:
        keys = (_PROMPT, _CACHED, _PREDICTED)
        if any(k not in before or k not in after for k in keys):
            return None, None, None
        prompt, cached, predicted = (round(after[k] - before[k]) for k in keys)
        if min(prompt, cached, predicted) < 0:
            self.negative = True
        return prompt + cached, cached, predicted

    async def request(
        self, phase: Phase, index: int, client: int, item_id: str, text: str
    ) -> RequestRecord:
        measured = phase in _MEASURED
        before: dict[str, float] = {}
        began = self.clock()
        if measured:
            before, busy = await self.wait_idle()
            if phase == "series" and index == 0 and not before:
                raise BenchUnavailable("llama-server /metrics is not reachable")
            if busy:
                return self.record(phase, index, client, item_id, began, None, "runtime_busy")
        sent = self.clock()
        status: int | None = None
        try:
            response = await self.http.post(
                f"{self.target.api_url}/classify",
                json={"message": text},
                headers=self.api_headers,
                timeout=self.client_timeout_s,
            )
        except httpx.TimeoutException:
            outcome = "client_timeout"
        except httpx.TransportError:
            outcome = "transport_error"
        else:
            status = response.status_code
            if status == 401 and phase == "warmup" and index == 0:
                raise BenchUnavailable("the API answered 401: export API_KEY before the run")
            if status == 200:
                outcome = "ok"
                if self.model is None:
                    self.model = _answer_model(response)
            else:
                outcome = _error_code(response) or f"http_{status}"
        wall_ms = round((self.clock() - sent) * 1000)
        counts = (None, None, None)
        if measured and before:
            counts = self.tokens(before, await self.metrics())
        return self.record(
            phase, index, client, item_id, sent, status, outcome, wall_ms=wall_ms, counts=counts
        )

    def record(
        self,
        phase: Phase,
        index: int,
        client: int,
        item_id: str,
        began: float,
        status: int | None,
        outcome: str,
        *,
        wall_ms: int | None = None,
        counts: tuple[int | None, int | None, int | None] = (None, None, None),
    ) -> RequestRecord:
        record = RequestRecord(
            phase=phase,
            index=index,
            client=client,
            item_id=item_id,
            started_s=began - self.start,
            status=status,
            outcome=outcome,
            wall_ms=round((self.clock() - began) * 1000) if wall_ms is None else wall_ms,
            prompt_tokens=counts[0],
            cached_tokens=counts[1],
            completion_tokens=counts[2],
        )
        self.add(record)
        return record


async def _cold_start(bench: _Bench) -> tuple[ColdStart, str | None, dict[str, str], str]:
    target = bench.target
    engine = target.engine
    files = [arg for f in target.compose_files for arg in ("-f", f)]
    compose = [engine, "compose", *files]
    down = bench.run([*compose, "down"])
    if down.code != 0:
        return ColdStart(health_s=None, ready_s=None), "cold_start_failed", {}, down.stderr_tail
    # A stack that survived compose down would make the cold start times meaningless.
    if await bench.get_ok(f"{target.api_url}/ready") or await bench.get_ok(
        f"{target.llm_url}/health"
    ):
        detail = "the old stack still answers after compose down"
        return ColdStart(health_s=None, ready_s=None), "cold_start_failed", {}, detail
    t0 = bench.clock()
    env = {**os.environ, "LLAMA_ARG_OFFLINE": "1"}
    up = asyncio.create_task(
        asyncio.to_thread(
            bench.run, [*compose, "up", "-d"], env=env, timeout_s=SLOW_COMMAND_TIMEOUT_S
        )
    )
    health_s: float | None = None
    ready_s: float | None = None
    while bench.clock() - t0 < COLD_START_MAX_S:
        if health_s is None and await bench.get_ok(f"{target.llm_url}/health"):
            health_s = bench.clock() - t0
        if ready_s is None and await bench.get_ok(f"{target.api_url}/ready"):
            ready_s = bench.clock() - t0
        if health_s is not None and ready_s is not None:
            break
        if up.done() and up.result().code != 0:
            break
        await bench.sleep(COLD_POLL_S)
    cold = ColdStart(health_s=health_s, ready_s=ready_s)
    started = await up
    if health_s is None or ready_s is None or started.code != 0:
        return cold, "cold_start_failed", {}, started.stderr_tail if up.done() else ""
    llm_env = read_env(bench.run, target.engine, target.llm_container)
    if llm_env is None or llm_env.get("LLAMA_ARG_OFFLINE") != "1":
        return cold, "not_offline", llm_env or {}, ""
    return cold, None, llm_env, ""


async def run_bench(
    target: Target,
    plan: Plan,
    *,
    run: Runner,
    http: httpx.AsyncClient,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_record: Callable[[RequestRecord], None] | None = None,
    on_phase: Callable[[Phase], None] | None = None,
    api_key: str | None = None,
) -> RunResult:
    bench = _Bench(target, run, http, clock, sleep, on_record, plan.max_s, api_key)
    cold: ColdStart | None = None
    llm_env: dict[str, str] = {}
    if plan.cold_start:
        cold, failure, llm_env, detail = await _cold_start(bench)
        if failure is not None:
            return RunResult([], [], cold, False, failure, None, {}, llm_env, detail)
    else:
        if not await bench.get_ok(f"{target.api_url}/ready"):
            raise BenchUnavailable(f"the API is not ready at {target.api_url}")
        llm_env = read_env(run, target.engine, target.llm_container) or {}
    bench.start = clock()

    api_env = read_env(run, target.engine, target.api_container)
    if api_env is None:
        raise BenchUnavailable(f"cannot read the env of container {target.api_container}")
    bench.client_timeout_s = (
        _seconds_setting(api_env, "QUEUE_TIMEOUT_S")
        + _seconds_setting(api_env, "REQUEST_TIMEOUT_S")
        + EXTRA_TIMEOUT_S
    )
    queue_size = (
        _count_setting(api_env, "MAX_CONCURRENCY") + _count_setting(api_env, "MAX_QUEUE") + 1
    )
    restarts0 = restart_count(run, target.engine, target.llm_container)

    inputs = plan.inputs
    phases: list[tuple[Phase, Callable[[], Awaitable[None]]]] = [
        ("warmup", lambda: _one_by_one(bench, "warmup", inputs[:WARMUP_ROWS])),
        ("series", lambda: _one_by_one(bench, "series", inputs)),
        ("long", lambda: _one_by_one(bench, "long", [("long", plan.long_text)])),
    ]
    if not plan.skip_sequential:
        cycle = [inputs[i % len(inputs)] for i in range(SEQUENTIAL_ROWS)]
        phases.append(("sequential", lambda: _one_by_one(bench, "sequential", cycle)))
    phases.append(("parallel", lambda: _parallel(bench, inputs)))
    phases.append(("queue", lambda: _queue(bench, inputs, queue_size)))

    stop: str | None = None
    for phase, body in phases:
        measured = phase != "warmup"
        if measured and bench.time_is_up():
            stop = "time_limit"
            break
        if on_phase is not None:
            on_phase(phase)
        if measured:
            await bench.take_reading(phase, "start")
        await body()
        if measured:
            await bench.take_reading(phase, "end")
        stop = _stop_after_phase(bench, restarts0) or ("time_limit" if bench.out_of_time else None)
        if stop is not None:
            break
    return RunResult(
        bench.records, bench.readings, cold, stop is None, stop, bench.model, api_env, llm_env
    )


def _stop_after_phase(bench: _Bench, restarts0: int | None) -> str | None:
    now = restart_count(bench.run, bench.target.engine, bench.target.llm_container)
    if restarts0 is not None and now is not None and now > restarts0:
        return "runtime_restarted"
    if bench.negative:
        return "negative_counter"
    return None


async def _one_by_one(bench: _Bench, phase: Phase, rows: Sequence[tuple[str, str]]) -> None:
    for index, (item_id, text) in enumerate(rows):
        if phase in _TIMED and bench.time_is_up():
            return
        await bench.request(phase, index, 0, item_id, text)


async def _parallel(bench: _Bench, inputs: Sequence[tuple[str, str]]) -> None:
    async def client(k: int) -> None:
        for index in range(k, len(inputs), CLIENTS):
            if bench.time_is_up():
                return
            item_id, text = inputs[index]
            await bench.request("parallel", index, k, item_id, text)

    await asyncio.gather(*(client(k) for k in range(CLIENTS)))


async def _queue(bench: _Bench, inputs: Sequence[tuple[str, str]], size: int) -> None:
    async def one(index: int) -> None:
        item_id, text = inputs[index % len(inputs)]
        await bench.request("queue", index, 0, item_id, text)

    await asyncio.gather(*(one(i) for i in range(size)))
