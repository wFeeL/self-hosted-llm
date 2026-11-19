import asyncio
import json
from collections.abc import Callable, Mapping, Sequence

import httpx
import respx

from error_classifier.benchmark.records import RequestRecord
from error_classifier.benchmark.runner import CommandResult, Plan, RunResult, Target, run_bench

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
MODEL_FILE = "/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf"


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
        self.api_auth: list[str | None] = []  # Authorization seen by /classify
        self.ready_auth: list[str | None] = []  # Authorization seen by /ready
        self.llm_auth: list[str | None] = []  # Authorization seen by llama-server
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
        self.up_stderr = ""
        self.down_code = 0
        self.down_stderr = ""
        self.build_stderr = ""
        self.unchecked = {
            "ready",
            "health",
        }  # first probe of each after down is the stale-stack check
        self.down_seen = False
        self.alive_after_down = False  # the old stack still answers after compose down
        self.api_env: dict[str, str] | None = dict(API_ENV)
        self.llm_env: dict[str, str] | None = dict(LLM_ENV)
        self.cgroup_code = 0
        self.build_code = 0
        self.cpu = {"llm_c": 0, "api_c": 0}
        self.docker_inspect = False  # {{.ImageName}} fails, the name is in {{.Config.Image}}
        router.get(f"{API}/ready").mock(side_effect=self._ready)
        router.get(f"{LLM}/health").mock(side_effect=self._health)
        router.get(f"{LLM}/metrics").mock(side_effect=self._metrics)
        router.get(f"{LLM}/props").mock(
            return_value=httpx.Response(200, json={"model_path": MODEL_FILE})
        )
        router.post(f"{API}/classify").mock(side_effect=self._classify)

    def _ok(self, n: int) -> httpx.Response:
        return httpx.Response(200, json={"category": "other", "model": "m-test"})

    def _ready(self, request: httpx.Request) -> httpx.Response:
        self.ready_auth.append(request.headers.get("authorization"))
        if "ready" in self.unchecked and self.down_seen:
            self.unchecked.discard("ready")
            return httpx.Response(200 if self.alive_after_down else 503)
        self.ready_calls += 1
        if self.ready_error is not None:
            raise self.ready_error
        status = 503 if self.ready_calls <= self.ready_after else 200
        return httpx.Response(status, json={"status": "ready"})

    def _health(self, request: httpx.Request) -> httpx.Response:
        self.llm_auth.append(request.headers.get("authorization"))
        if "health" in self.unchecked and self.down_seen:
            self.unchecked.discard("health")
            return httpx.Response(200 if self.alive_after_down else 503)
        self.health_calls += 1
        return httpx.Response(503 if self.health_calls <= self.health_after else 200)

    def _metrics(self, request: httpx.Request) -> httpx.Response:
        self.llm_auth.append(request.headers.get("authorization"))
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
        self.api_auth.append(request.headers.get("authorization"))
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
            if "version" in args:
                return CommandResult(0, "podman-compose version 1.5.0\nsecond line\n")
            if "down" in args:
                self.down_seen = True
                return CommandResult(self.down_code, "", self.down_stderr)
            if "up" in args:
                return CommandResult(self.up_code, "", self.up_stderr)
            if "build" in args:
                return CommandResult(self.build_code, "", self.build_stderr)
            return CommandResult(0, "")
        if engine_command == "ps":
            return CommandResult(0, "hawkstand_a\nllm_c\nhawkstand_b\n")
        if engine_command == "image":
            return CommandResult(0, "123456789\n")
        container = args[-1]
        if engine_command == "inspect":
            template = args[3]
            if template == "{{.RestartCount}}":
                return CommandResult(0, f"{self.restarts}\n")
            if template == "{{.ImageName}}":
                if self.docker_inspect:
                    return CommandResult(1, "")
                return CommandResult(0, f"img-{container}:tag\n")
            if template == "{{.Config.Image}}":
                if self.docker_inspect:
                    return CommandResult(0, f"cfg-{container}:tag\n")
                return CommandResult(1, "")
            if template == "{{.Image}}":
                return CommandResult(0, f"sha-{container}\n")
            env_map = self.api_env if container == "api_c" else self.llm_env
            if env_map is None:
                return CommandResult(1, "")
            return CommandResult(0, json.dumps([f"{k}={v}" for k, v in env_map.items()]))
        if engine_command == "exec":
            if args[3] == "stat":
                return CommandResult(0, "2097152\n")
            return self._cat(args[2], args[-1].rsplit("/", 1)[-1])
        return CommandResult(1, "")

    def _cat(self, container: str, name: str) -> CommandResult:
        if self.cgroup_code:
            return CommandResult(self.cgroup_code, "")
        if name == "cpu.stat":
            self.cpu[container] += 1_000_000
            return CommandResult(0, f"usage_usec {self.cpu[container]}\nuser_usec 5\n")
        if name == "memory.stat":
            stat = {"llm_c": "anon 3000\nfile 700\n", "api_c": "anon 300\nfile 70\n"}
            return CommandResult(0, stat[container])
        # Different per container and file, so a swapped container or file name fails a test.
        values = {
            "memory.current": {"llm_c": "5000\n", "api_c": "500\n"},
            "memory.peak": {"llm_c": "9000\n", "api_c": "900\n"},
            "memory.swap.current": {"llm_c": "40\n", "api_c": "4\n"},
            "memory.swap.peak": {"llm_c": "70\n", "api_c": "7\n"},
            "memory.swap.max": {"llm_c": "0\n", "api_c": "max\n"},
            "memory.max": {"llm_c": "6442450944\n", "api_c": "536870912\n"},
            "cpu.max": {"llm_c": "400000 100000\n", "api_c": "max 100000\n"},
        }
        value = values.get(name, {}).get(container)
        return CommandResult(1, "") if value is None else CommandResult(0, value)

    async def run(
        self,
        plan: Plan | None = None,
        *,
        on_record: Callable[[RequestRecord], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
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
                on_phase=on_phase,
            )
