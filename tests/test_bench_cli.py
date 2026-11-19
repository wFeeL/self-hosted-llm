import hashlib
import json
from pathlib import Path

import httpx
import pytest
from bench_fakes import API, API_ENV, LLM, LLM_ENV, MODEL_FILE, Stack
from bench_files import EXPECTED, alpha, beta

from error_classifier.benchmark import cli
from error_classifier.benchmark.aggregate import aggregate
from error_classifier.benchmark.report import read_bench, write_bench
from error_classifier.benchmark.runner import bench_inputs, long_input
from error_classifier.evaluation.dataset import DEFAULT_DATASET, load_dataset

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["podman", "compose", "-f", "compose.yaml", "-f", "compose.debug.yaml"]


class Env:
    def __init__(self, loads: list[float]) -> None:
        self.loads = loads
        self.pauses: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.pauses.append(seconds)

    def loadavg(self) -> tuple[float, float, float]:
        value = self.loads.pop(0) if len(self.loads) > 1 else self.loads[0]
        return (value, value, value)


@pytest.fixture(autouse=True)
def in_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def run_cli(
    stack: Stack, out_dir: Path, *extra: str, loads: list[float] | None = None
) -> tuple[int, Env]:
    env = Env(loads or [0.5])
    args = [
        "run",
        "--engine",
        "podman",
        "--llm-container",
        "llm_c",
        "--api-container",
        "api_c",
        "--api-url",
        API,
        "--llm-url",
        LLM,
        "--out-dir",
        str(out_dir),
        *extra,
    ]
    code = cli.main(
        args,
        sleep=env.sleep,
        loadavg=env.loadavg,
        run=stack.runner,
        clock=stack.clock,
        sleep_async=stack.clock.sleep,
    )
    return code, env


def only_file(out_dir: Path) -> Path:
    files = list(out_dir.iterdir())
    assert len(files) == 1
    return files[0]


def test_run_writes_a_complete_file(
    stack: Stack, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stack.llm_env = {**LLM_ENV, "PATH": "/usr/bin"}
    stack.api_env = {**API_ENV, "SECRET": "x", "MAX_OUTPUT_TOKENS": "256"}
    code, _ = run_cli(stack, tmp_path / "out")

    assert code == 0
    path = only_file(tmp_path / "out")
    bench = read_bench(path)
    config = bench.summary.config
    assert path.name == f"{config.started_at[:10]}-m-test.jsonl"
    assert (config.model, config.engine, config.complete, config.stop_reason) == (
        "m-test",
        "podman",
        True,
        None,
    )
    assert config.compose_version == "podman-compose version 1.5.0"
    assert config.compose_files == ["compose.yaml", "compose.debug.yaml"]
    assert (config.llm_container, config.api_container) == ("llm_c", "api_c")
    assert (config.llm_image, config.llm_image_id, config.llm_image_size) == (
        "img-llm_c:tag",
        "sha-llm_c",
        123456789,
    )
    assert (config.api_image, config.api_image_id, config.api_image_size) == (
        "img-api_c:tag",
        "sha-api_c",
        123456789,
    )
    assert config.runtime_env == LLM_ENV
    assert config.api_env == {**API_ENV, "MAX_OUTPUT_TOKENS": "256"}
    assert (config.quant, config.model_path, config.model_file_size) == (
        "Q4_K_M",
        MODEL_FILE,
        2097152,
    )
    inputs = bench_inputs(load_dataset(ROOT / DEFAULT_DATASET))
    expected_hash = hashlib.sha256(json.dumps([*inputs, long_input()]).encode()).hexdigest()
    assert config.inputs_sha256 == expected_hash
    assert config.phases == {
        "warmup": 3,
        "series": 20,
        "long": 1,
        "sequential": 50,
        "parallel": 20,
        "queue": 7,
    }
    assert (config.skip_sequential, config.cold_start) == (False, False)
    assert config.loadavg_before == [0.5, 0.5, 0.5]
    assert config.loadavg_after == [0.5, 0.5, 0.5]
    assert config.loaded is False
    assert config.hawk_containers == 2
    assert config.machine
    assert config.cpu_count
    assert bench.summary.cold_start is None
    memory = bench.summary.memory
    assert (memory.llm_peak, memory.llm_anon, memory.llm_file) == (9000, 3000, 700)
    assert (memory.api_peak, memory.api_anon, memory.api_file) == (900, 300, 70)
    assert (memory.llm_swap_peak, memory.llm_swap_max, memory.llm_memory_max) == (
        70,
        "0",
        "6442450944",
    )
    assert (memory.api_swap_peak, memory.api_swap_max, memory.api_memory_max) == (
        7,
        "max",
        "536870912",
    )
    assert bench.summary.aggregates == aggregate(bench.records, bench.readings)
    assert bench.summary.aggregates.series.n_ok == 20
    assert len(bench.records) == 101
    err = capsys.readouterr().err
    assert "phase series" in err
    assert "series 0 ok 0.1s" in err
    assert "queue 6 ok" in err


def test_run_records_the_cpu_limit(stack: Stack, tmp_path: Path) -> None:
    code, _ = run_cli(stack, tmp_path)

    assert code == 0
    assert read_bench(only_file(tmp_path)).summary.config.llm_cpu_max == "400000 100000"


def test_waiting_for_load_is_recorded(stack: Stack, tmp_path: Path) -> None:
    code, env = run_cli(stack, tmp_path, loads=[2.0, 0.5])

    assert code == 0
    assert env.pauses == [15]
    config = read_bench(only_file(tmp_path)).summary.config
    assert config.loadavg_before == [0.5, 0.5, 0.5]
    assert config.loaded is False


def test_no_load_wait_skips_the_wait(stack: Stack, tmp_path: Path) -> None:
    code, env = run_cli(stack, tmp_path, "--no-load-wait", loads=[3.0])

    assert code == 0
    assert env.pauses == []
    config = read_bench(only_file(tmp_path)).summary.config
    assert config.loadavg_before == [3.0, 3.0, 3.0]
    assert config.loaded is True


def test_busy_machine_marks_the_file_loaded(stack: Stack, tmp_path: Path) -> None:
    code, env = run_cli(stack, tmp_path, loads=[2.0])

    assert code == 0
    assert sum(env.pauses) == 600
    config = read_bench(only_file(tmp_path)).summary.config
    assert config.loadavg_before == [2.0, 2.0, 2.0]
    assert config.loaded is True


def test_skip_sequential_and_custom_compose_files(stack: Stack, tmp_path: Path) -> None:
    code, _ = run_cli(stack, tmp_path, "--skip-sequential", "--compose-file", "other.yaml")

    assert code == 0
    config = read_bench(only_file(tmp_path)).summary.config
    assert config.skip_sequential is True
    assert "sequential" not in config.phases
    assert config.compose_files == ["other.yaml"]


def test_cold_start_builds_then_restarts_the_stack(stack: Stack, tmp_path: Path) -> None:
    code, _ = run_cli(stack, tmp_path, "--cold-start")

    assert code == 0
    compose = [c.args for c in stack.calls if c.args[1] == "compose" and "version" not in c.args]
    assert compose == [[*COMPOSE, "build", "api"], [*COMPOSE, "down"], [*COMPOSE, "up", "-d"]]
    bench = read_bench(only_file(tmp_path))
    assert bench.summary.config.cold_start is True
    assert bench.summary.cold_start is not None
    assert bench.summary.cold_start.ready_s == 0


def test_failed_build_stops_before_the_stack_is_touched(
    stack: Stack, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stack.build_code = 1
    stack.build_stderr = "pip: no network"
    code, _ = run_cli(stack, tmp_path / "out", "--cold-start")

    assert code == 3
    assert "pip: no network" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    assert all("down" not in c.args for c in stack.calls)


def test_failed_cold_start_prints_the_error_tail(
    stack: Stack, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stack.down_code = 1
    stack.down_stderr = "no such pod"
    code, _ = run_cli(stack, tmp_path / "out", "--cold-start")

    assert code == 4
    err = capsys.readouterr().err
    assert "run stopped: cold_start_failed" in err
    assert "no such pod" in err


def test_unreachable_api_gives_code_3_and_no_file(
    stack: Stack, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stack.ready_error = httpx.ConnectError("refused")
    code, _ = run_cli(stack, tmp_path / "out")

    assert code == 3
    assert not (tmp_path / "out").exists()
    assert "not ready" in capsys.readouterr().err


def test_missing_metrics_gives_code_3(stack: Stack, tmp_path: Path) -> None:
    stack.metrics_status = 404
    code, _ = run_cli(stack, tmp_path / "out")

    assert code == 3
    assert not (tmp_path / "out").exists()


def test_time_limit_writes_a_partial_file(stack: Stack, tmp_path: Path) -> None:
    code, _ = run_cli(stack, tmp_path, "--max-minutes", "0.0001")

    assert code == 4
    path = only_file(tmp_path)
    assert path.name.endswith(".jsonl.partial")
    config = read_bench(path).summary.config
    assert (config.complete, config.stop_reason) == (False, "time_limit")


def test_unreadable_dataset_gives_code_2(stack: Stack, tmp_path: Path) -> None:
    code, _ = run_cli(stack, tmp_path, "--dataset", str(tmp_path / "missing.jsonl"))

    assert code == 2
    assert stack.posts == []


def test_run_needs_an_engine() -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(["run", "--llm-container", "a", "--api-container", "b"])
    assert stop.value.code == 2


def write_two(tmp_path: Path) -> list[Path]:
    paths: list[Path] = []
    for bench in (alpha(), beta()):
        path = tmp_path / f"{bench.summary.config.model}.jsonl"
        write_bench(path, bench.records, bench.readings, bench.summary)
        paths.append(path)
    return paths


def test_report_prints_the_tables(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = write_two(tmp_path)
    code = cli.main(["report", *map(str, reversed(paths))])

    assert code == 0
    assert capsys.readouterr().out == EXPECTED


def test_report_refuses_a_partial_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first, second = write_two(tmp_path)
    broken = read_bench(second).summary
    broken = broken.model_copy(
        update={"config": broken.config.model_copy(update={"complete": False})}
    )
    write_bench(second, [], [], broken)

    assert cli.main(["report", str(first), str(second)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "incomplete" in captured.err


def test_report_refuses_unreadable_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junk = tmp_path / "junk.jsonl"
    junk.write_text("not json\n", encoding="utf-8")

    assert cli.main(["report", str(junk)]) == 2
    assert cli.main(["report", str(tmp_path / "missing.jsonl")]) == 2
    assert "junk.jsonl" in capsys.readouterr().err
