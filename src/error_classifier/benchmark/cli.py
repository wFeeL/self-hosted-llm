import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

import httpx

from error_classifier.benchmark.aggregate import aggregate
from error_classifier.benchmark.records import (
    Phase,
    RequestRecord,
    parse_int,
    parse_memory_stat,
    quant_of,
)
from error_classifier.benchmark.report import (
    BenchConfig,
    BenchFile,
    Memory,
    Summary,
    read_bench,
    render_bench_report,
    write_bench,
)
from error_classifier.benchmark.runner import (
    SEQUENTIAL_ROWS,
    SLOW_COMMAND_TIMEOUT_S,
    WARMUP_ROWS,
    BenchUnavailable,
    Plan,
    Runner,
    RunResult,
    Target,
    bench_inputs,
    cgroup_file,
    long_input,
    run_bench,
    subprocess_runner,
)
from error_classifier.evaluation.dataset import DEFAULT_DATASET, DatasetError, load_dataset
from error_classifier.evaluation.load import LoadAvg, wait_for_load
from error_classifier.evaluation.metrics import LOAD_LIMIT
from error_classifier.evaluation.results import git_state

DEFAULT_COMPOSE_FILES: Final = ("compose.yaml", "compose.debug.yaml")
API_ENV_KEYS: Final = (
    "LLM_MODEL",
    "MAX_CONCURRENCY",
    "MAX_QUEUE",
    "QUEUE_TIMEOUT_S",
    "REQUEST_TIMEOUT_S",
    "MAX_OUTPUT_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_DISABLE_THINKING",
)
HAWK_PREFIX: Final = "hawkstand_"


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the benchmark protocol on a running stack")
    run.add_argument("--engine", choices=("podman", "docker"), required=True)
    run.add_argument("--llm-container", required=True)
    run.add_argument("--api-container", required=True)
    run.add_argument("--api-url", default="http://127.0.0.1:8000")
    run.add_argument("--llm-url", default="http://127.0.0.1:8080")
    # No default here: argparse appends to a default list instead of replacing it.
    run.add_argument("--compose-file", action="append", default=None)
    run.add_argument("--cold-start", action="store_true")
    run.add_argument("--skip-sequential", action="store_true")
    run.add_argument("--max-minutes", type=float, default=15.0)
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--out-dir", type=Path, default=Path("bench/results"))
    report = commands.add_parser("report", help="print tables for benchmark files")
    report.add_argument("paths", type=Path, nargs="+", metavar="PATH")
    return parser


def _progress(record: RequestRecord) -> None:
    _err(f"{record.phase} {record.index} {record.outcome} {record.wall_ms / 1000:.1f}s")


def _phase_started(phase: Phase) -> None:
    _err(f"phase {phase}")


def _first_line(text: str) -> str | None:
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else None


def _image(run: Runner, engine: str, container: str) -> tuple[str | None, str | None, int | None]:
    # {{.ImageName}} is podman only, Docker keeps the name in .Config.Image.
    name = run([engine, "inspect", "--format", "{{.ImageName}}", container])
    text = name.stdout.strip() if name.code == 0 else ""
    if not text:
        fallback = run([engine, "inspect", "--format", "{{.Config.Image}}", container])
        text = fallback.stdout.strip() if fallback.code == 0 else ""
    ident = run([engine, "inspect", "--format", "{{.Image}}", container])
    image_id = ident.stdout.strip() if ident.code == 0 and ident.stdout.strip() else None
    size: int | None = None
    if image_id is not None:
        found = run([engine, "image", "inspect", "--format", "{{.Size}}", image_id])
        size = parse_int(found.stdout) if found.code == 0 else None
    return text or None, image_id, size


def _cgroup_memory(run: Runner, engine: str, container: str) -> tuple[int | None, dict[str, int]]:
    peak = cgroup_file(run, engine, container, "memory.peak")
    stat = cgroup_file(run, engine, container, "memory.stat")
    return (
        None if peak is None else parse_int(peak),
        {} if stat is None else parse_memory_stat(stat),
    )


async def _model_path(http: httpx.AsyncClient, llm_url: str) -> str | None:
    try:
        response = await http.get(f"{llm_url}/props", timeout=2.0)
        body = cast(object, response.json()) if response.status_code == 200 else None
    except httpx.HTTPError, ValueError:
        body = None
    if not isinstance(body, dict):
        _err("warning: /props unavailable, model_path left empty")
        return None
    path = cast("dict[str, object]", body).get("model_path")
    return path if isinstance(path, str) else None


def _queue_size(api_env: dict[str, str]) -> int:
    try:
        return int(api_env["MAX_CONCURRENCY"]) + int(api_env["MAX_QUEUE"]) + 1
    except KeyError, ValueError:
        return 0


def _planned(plan: Plan, queue_size: int) -> dict[str, int]:
    phases = {"warmup": WARMUP_ROWS, "series": len(plan.inputs), "long": 1}
    if not plan.skip_sequential:
        phases["sequential"] = SEQUENTIAL_ROWS
    phases["parallel"] = len(plan.inputs)
    phases["queue"] = queue_size
    return phases


def _hawk_containers(run: Runner, engine: str) -> int | None:
    listing = run([engine, "ps", "--format", "{{.Names}}"])
    if listing.code != 0:
        return None
    return sum(1 for name in listing.stdout.split() if name.startswith(HAWK_PREFIX))


def _rounded(load: tuple[float, float, float]) -> list[float]:
    return [round(x, 2) for x in load]


async def _execute(
    target: Target,
    plan: Plan,
    run: Runner,
    clock: Callable[[], float],
    sleep_async: Callable[[float], Awaitable[None]],
) -> tuple[RunResult, str | None]:
    async with httpx.AsyncClient() as http:
        result = await run_bench(
            target,
            plan,
            run=run,
            http=http,
            clock=clock,
            sleep=sleep_async,
            on_record=_progress,
            on_phase=_phase_started,
        )
        return result, await _model_path(http, target.llm_url)


def _run(
    args: argparse.Namespace,
    sleep: Callable[[float], None],
    loadavg: LoadAvg,
    run: Runner,
    clock: Callable[[], float],
    sleep_async: Callable[[float], Awaitable[None]],
) -> int:
    engine = cast(Literal["podman", "docker"], args.engine)
    files = tuple(cast("list[str] | None", args.compose_file) or DEFAULT_COMPOSE_FILES)
    try:
        inputs = bench_inputs(load_dataset(cast(Path, args.dataset)))
    except (DatasetError, OSError, ValueError) as exc:
        _err(str(exc))
        return 2
    before = wait_for_load(sleep, loadavg)
    cold_start = cast(bool, args.cold_start)
    if cold_start:
        compose = [engine, "compose", *(a for f in files for a in ("-f", f))]
        built = run([*compose, "build", "api"], timeout_s=SLOW_COMMAND_TIMEOUT_S)
        if built.code != 0:
            _err(f"compose build api failed: {built.stderr_tail.strip()}")
            return 3
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_sha, git_dirty = git_state(Path.cwd())
    llm_container = cast(str, args.llm_container)
    api_container = cast(str, args.api_container)
    target = Target(
        engine=engine,
        api_url=cast(str, args.api_url),
        llm_url=cast(str, args.llm_url),
        llm_container=llm_container,
        api_container=api_container,
        compose_files=files,
    )
    plan = Plan(
        inputs=inputs,
        long_text=long_input(),
        cold_start=cold_start,
        skip_sequential=cast(bool, args.skip_sequential),
        max_s=cast(float, args.max_minutes) * 60,
    )
    try:
        result, model_path = asyncio.run(_execute(target, plan, run, clock, sleep_async))
    except BenchUnavailable as exc:
        _err(str(exc))
        return 3
    after = loadavg()

    llm_peak, llm_stat = _cgroup_memory(run, engine, llm_container)
    api_peak, api_stat = _cgroup_memory(run, engine, api_container)
    llm_image, llm_image_id, llm_image_size = _image(run, engine, llm_container)
    api_image, api_image_id, api_image_size = _image(run, engine, api_container)
    model_file_size: int | None = None
    if model_path is not None:
        sized = run([engine, "exec", llm_container, "stat", "-L", "-c", "%s", model_path])
        model_file_size = parse_int(sized.stdout) if sized.code == 0 else None
    version = run([engine, "compose", "version"])
    runtime_env = {k: v for k, v in result.llm_env.items() if k.startswith("LLAMA_ARG_")}
    quant_source = Path(model_path).name if model_path else None
    model = result.model or result.api_env.get("LLM_MODEL") or "unknown"
    config = BenchConfig(
        model=model,
        engine=engine,
        compose_version=_first_line(version.stdout) if version.code == 0 else None,
        compose_files=list(files),
        llm_container=llm_container,
        api_container=api_container,
        llm_image=llm_image,
        llm_image_id=llm_image_id,
        llm_image_size=llm_image_size,
        api_image=api_image,
        api_image_id=api_image_id,
        api_image_size=api_image_size,
        runtime_env=runtime_env,
        api_env={k: result.api_env[k] for k in API_ENV_KEYS if k in result.api_env},
        quant=quant_of(quant_source or "") or quant_of(runtime_env.get("LLAMA_ARG_HF_FILE", "")),
        model_path=model_path,
        model_file_size=model_file_size,
        inputs_sha256=hashlib.sha256(
            json.dumps([*plan.inputs, plan.long_text]).encode()
        ).hexdigest(),
        phases=_planned(plan, _queue_size(result.api_env)),
        skip_sequential=plan.skip_sequential,
        cold_start=cold_start,
        git_sha=git_sha,
        git_dirty=git_dirty,
        started_at=started_at,
        loadavg_before=_rounded(before),
        loadavg_after=_rounded(after),
        loaded=before[0] >= LOAD_LIMIT,
        hawk_containers=_hawk_containers(run, engine),
        cpu_count=os.cpu_count(),
        machine=platform.machine(),
        complete=result.complete,
        stop_reason=result.stop_reason,
    )
    summary = Summary(
        config=config,
        cold_start=result.cold_start,
        memory=Memory(
            llm_peak=llm_peak,
            llm_anon=llm_stat.get("anon"),
            llm_file=llm_stat.get("file"),
            api_peak=api_peak,
            api_anon=api_stat.get("anon"),
        ),
        aggregates=aggregate(result.records, result.readings),
    )
    name = f"{started_at[:10]}-{model}.jsonl"
    path = cast(Path, args.out_dir) / (name if result.complete else name + ".partial")
    write_bench(path, result.records, result.readings, summary)
    if result.complete:
        return 0
    _err(f"run stopped: {result.stop_reason}, wrote {path}")
    if result.failure_detail:
        _err(result.failure_detail.strip())
    return 4


def _report(args: argparse.Namespace) -> int:
    files: list[BenchFile] = []
    for path in cast("list[Path]", args.paths):
        try:
            files.append(read_bench(path))
        except (OSError, ValueError) as exc:
            _err(f"{path}: {exc}")
            return 2
    try:
        text = render_bench_report(files)
    except ValueError as exc:
        _err(str(exc))
        return 2
    sys.stdout.write(text)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    loadavg: LoadAvg = os.getloadavg,
    run: Runner = subprocess_runner,
    clock: Callable[[], float] = time.perf_counter,
    sleep_async: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args, sleep, loadavg, run, clock, sleep_async)
    return _report(args)
