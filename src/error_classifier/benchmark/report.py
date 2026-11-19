import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict

from error_classifier.benchmark.aggregate import Aggregates
from error_classifier.benchmark.records import Reading, RequestRecord
from error_classifier.benchmark.runner import ColdStart

# Models differ in these keys on purpose, everything else must match.
MODEL_RUNTIME_KEYS: Final = frozenset({"LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_FILE", "LLAMA_ARG_ALIAS"})
MODEL_API_KEYS: Final = frozenset({"LLM_MODEL", "LLM_DISABLE_THINKING"})
GIB: Final = 2**30


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchConfig(_Model):
    model: str  # "model" of the first ok answer, else LLM_MODEL
    engine: Literal["podman", "docker"]
    compose_version: str | None  # first line of `<engine> compose version`
    compose_files: list[str]
    llm_container: str
    api_container: str
    llm_image: str | None
    llm_image_id: str | None
    llm_image_size: int | None
    api_image: str | None
    api_image_id: str | None
    api_image_size: int | None
    runtime_env: dict[str, str]  # LLAMA_ARG_* of llm
    api_env: dict[str, str]  # D9 keys of api
    quant: str | None
    model_path: str | None  # /props
    model_file_size: int | None
    inputs_sha256: str
    phases: dict[str, int]  # planned requests per phase
    skip_sequential: bool
    cold_start: bool
    git_sha: str | None
    git_dirty: bool | None
    started_at: str
    loadavg_before: list[float] | None
    loadavg_after: list[float] | None
    loaded: bool
    hawk_containers: int | None
    cpu_count: int | None
    machine: str
    llm_cpu_max: str | None = None  # absent in files written before the limit was read
    complete: bool
    stop_reason: str | None


class Memory(_Model):
    llm_peak: int | None
    llm_anon: int | None
    llm_file: int | None
    api_peak: int | None
    api_anon: int | None
    api_file: int | None = None  # absent in the first three result files
    # The fields below are absent in files written before swap and limits were read.
    llm_swap_peak: int | None = None
    api_swap_peak: int | None = None
    llm_swap_max: str | None = None
    api_swap_max: str | None = None
    llm_memory_max: str | None = None
    api_memory_max: str | None = None


class Summary(_Model):
    type: Literal["summary"] = "summary"
    config: BenchConfig
    cold_start: ColdStart | None
    memory: Memory
    aggregates: Aggregates


@dataclass(frozen=True)
class BenchFile:
    records: list[RequestRecord]
    readings: list[Reading]
    summary: Summary


def write_bench(
    path: Path, records: Sequence[RequestRecord], readings: Sequence[Reading], summary: Summary
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.model_dump_json() for r in records]
    lines += [r.model_dump_json() for r in readings]
    lines.append(summary.model_dump_json())
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_bench(path: Path) -> BenchFile:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("the file is empty")
    records: list[RequestRecord] = []
    readings: list[Reading] = []
    summary: Summary | None = None
    for number, line in enumerate(lines, start=1):
        if summary is not None:
            raise ValueError(f"line {number}: rows after the summary")
        data = cast(object, json.loads(line))
        kind = cast("dict[str, object]", data).get("type") if isinstance(data, dict) else None
        if kind == "request":
            records.append(RequestRecord.model_validate(data))
        elif kind == "reading":
            readings.append(Reading.model_validate(data))
        elif kind == "summary":
            summary = Summary.model_validate(data)
        else:
            raise ValueError(f"line {number}: unknown row type {kind!r}")
    if summary is None:
        raise ValueError("the last line must be the summary")
    return BenchFile(records, readings, summary)


def _line(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    separator = "|" + "---|" * len(header)
    return "\n".join([_line(header), separator, *(_line(row) for row in rows)])


def _seconds(ms: int | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}"


def _float(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _dash(value: str | None) -> str:
    return "-" if value is None else value


def _gib(size: int | None) -> str:
    return "-" if size is None else f"{size / GIB:.2f} GiB"


def _total(*parts: int | None) -> int | None:
    return None if any(p is None for p in parts) else sum(cast("list[int]", parts))


def llm_need_bytes(bench: BenchFile) -> int | None:
    summary = bench.summary
    conservative = _total(summary.aggregates.llm_anon_max, summary.config.model_file_size)
    peak = summary.memory.llm_peak
    return None if conservative is None or peak is None else max(conservative, peak)


def cpu_limit(raw: str) -> str:
    parts = raw.split()
    if len(parts) != 2:
        return raw
    quota, period = parts
    if quota == "max":
        return "none"
    try:
        quota_us, period_us = int(quota), int(period)
    except ValueError:
        return raw
    if period_us <= 0:
        return raw
    return f"{round(quota_us / period_us, 2):g}"


def _same(files: Sequence[BenchFile], what: str, key: Callable[[BenchConfig], object]) -> None:
    if len({json.dumps(key(f.summary.config), sort_keys=True) for f in files}) != 1:
        raise ValueError(f"the results come from different {what}")


def _check_rows(f: BenchFile) -> None:
    counted = Counter(r.phase for r in f.records)
    planned = f.summary.config.phases
    for phase in sorted(planned.keys() | counted.keys()):
        if counted.get(phase, 0) != planned.get(phase, 0):
            raise ValueError(
                f"{f.summary.config.model}: phase {phase} has {counted.get(phase, 0)} "
                f"records, the plan says {planned.get(phase, 0)}"
            )


def _check(files: Sequence[BenchFile]) -> None:
    if not files:
        raise ValueError("no results to report")
    for f in files:
        if not f.summary.config.complete:
            raise ValueError(f"{f.summary.config.model}: the run is incomplete")
    _same(files, "inputs", lambda c: c.inputs_sha256)
    _same(files, "engines", lambda c: c.engine)
    # The API image differs on purpose: it is built from the package that holds this script.
    _same(files, "llm image id", lambda c: c.llm_image_id)
    _same(files, "cpu count", lambda c: c.cpu_count)
    _same(files, "machine", lambda c: c.machine)
    _same(files, "llm CPU limits", lambda c: c.llm_cpu_max)
    _same(files, "compose version", lambda c: c.compose_version)
    for f in files:
        _check_rows(f)
    for name, setting, own in (
        ("runtime", "runtime_env", MODEL_RUNTIME_KEYS),
        ("API", "api_env", MODEL_API_KEYS),
    ):
        envs = [getattr(f.summary.config, setting) for f in files]
        keys = sorted({k for env in envs for k in env} - own)
        for key in keys:
            if len({env.get(key) for env in envs}) != 1:
                raise ValueError(f"the results come from different {name} settings: {key}")


def _hardware(config: BenchConfig) -> str:
    env = config.runtime_env
    limit = "" if config.llm_cpu_max is None else f" llm CPU limit {cpu_limit(config.llm_cpu_max)};"
    return (
        f"Engine: {config.engine}, {config.compose_version or '-'}; "
        f"{config.cpu_count if config.cpu_count is not None else '-'} vCPU {config.machine};"
        f"{limit} "
        f"llm threads {env.get('LLAMA_ARG_THREADS', '-')}, "
        f"context {env.get('LLAMA_ARG_CTX_SIZE', '-')}, "
        f"slots {env.get('LLAMA_ARG_N_PARALLEL', '-')}, "
        f"batch {env.get('LLAMA_ARG_BATCH', '-')}."
    )


def _load(config: BenchConfig) -> str:
    if not config.loadavg_before:
        return "-"
    text = f"{config.loadavg_before[0]:.2f}"
    return f"{text} loaded" if config.loaded else text


def render_bench_report(files: Sequence[BenchFile]) -> str:
    _check(files)
    ordered = sorted(files, key=lambda f: f.summary.config.model)

    speed_rows: list[list[str]] = []
    load_rows: list[list[str]] = []
    drift_rows: list[list[str]] = []
    memory_rows: list[list[str]] = []
    need_rows: list[list[str]] = []
    for f in ordered:
        s = f.summary
        c, a, m = s.config, s.aggregates, s.memory
        cold = s.cold_start
        series_cpu = a.cpu.get("series")
        non_ok = sum(n for outcome, n in a.series.outcomes.items() if outcome != "ok")
        speed_rows.append(
            [
                c.model,
                c.quant or "-",
                _gib(c.model_file_size),
                _load(c),
                _float(cold.health_s) if cold else "-",
                _float(cold.ready_s) if cold else "-",
                _seconds(a.series.p50_ms),
                _seconds(a.series.p95_ms),
                _seconds(a.series.max_ms),
                str(a.series.n_ok),
                str(non_ok),
                _float(a.speed.generation_tps),
                _float(a.speed.prompt_tps),
                _float(series_cpu.llm_percent) if series_cpu else "-",
            ]
        )
        parallel_non_ok = sum(n for o, n in a.parallel.outcomes.items() if o != "ok")
        load_rows.append(
            [
                c.model,
                _seconds(a.long_input.p50_ms),
                ", ".join(sorted(a.long_input.outcomes)) or "-",
                _seconds(a.parallel.p95_ms),
                str(parallel_non_ok),
                str(a.queue.immediate_429),
                str(a.queue.timeout_429),
                str(sum(n for o, n in a.queue.outcomes.items() if o not in ("ok", "queue_full"))),
            ]
        )
        d = a.sequential
        every = a.sequential_all
        sequential_non_ok = (
            "-" if every is None else str(sum(n for o, n in every.outcomes.items() if o != "ok"))
        )
        drift_rows.append(
            [c.model, "-", "-", "-", "-", "-", "-", "-", "-"]
            if d is None or every is None
            else [
                c.model,
                _seconds(d.first10.p50_ms),
                _seconds(d.first10.p95_ms),
                _seconds(d.last10.p50_ms),
                _seconds(d.last10.p95_ms),
                _gib(d.memory_before),
                _gib(d.memory_after),
                str(every.n_ok),
                sequential_non_ok,
            ]
        )
        memory_rows.append(
            [
                c.model,
                _gib(m.llm_peak),
                _gib(m.llm_file),
                _gib(m.api_peak),
                _gib(m.api_file),
                _gib(m.llm_swap_peak),
                _dash(m.llm_swap_max),
                _dash(m.llm_memory_max),
                _gib(m.api_swap_peak),
                _dash(m.api_swap_max),
                _dash(m.api_memory_max),
                _gib(c.model_file_size),
                _gib(a.llm_anon_max),
                _gib(_total(a.llm_anon_max, c.model_file_size)),
                _gib(_total(m.llm_peak, c.model_file_size)),
                _gib(c.llm_image_size),
                _gib(c.api_image_size),
            ]
        )
        need = llm_need_bytes(f)
        need_rows.append(
            [
                c.model,
                _gib(need),
                _gib(m.api_peak),
                _gib(_total(c.llm_image_size, c.api_image_size, c.model_file_size)),
            ]
        )
    parts = [
        _hardware(ordered[0].summary.config),
        _table(
            [
                "Model",
                "Quant",
                "GGUF",
                "Load at start",
                "Cold start to /health, s",
                "Cold start to /ready, s",
                "p50, s",
                "p95, s",
                "Max, s",
                "n ok",
                "Non-ok",
                "Generation, tok/s",
                "Prompt, tok/s",
                "CPU llm, %",
            ],
            speed_rows,
        ),
        _table(
            [
                "Model",
                "Long input, s",
                "Long input outcome",
                "Two clients p95, s",
                "Two clients non-ok",
                "Queue probe: immediate 429",
                "Queue probe: 429 after wait",
                "Queue probe: other non-ok",
            ],
            load_rows,
        ),
        _table(
            [
                "Model",
                "p50 first 10, s",
                "p95 first 10, s",
                "p50 last 10, s",
                "p95 last 10, s",
                "Memory before",
                "Memory after",
                "n ok",
                "Non-ok",
            ],
            drift_rows,
        ),
        _table(
            [
                "Model",
                "llm peak",
                "llm file",
                "api peak",
                "api file",
                "llm swap peak",
                "llm swap max",
                "llm memory max",
                "api swap peak",
                "api swap max",
                "api memory max",
                "GGUF",
                "RAM floor",
                "RAM, conservative",
                "RAM, worst",
                "llm image",
                "api image",
            ],
            memory_rows,
        ),
        _table(["Model", "llm need", "api peak", "Disk"], need_rows),
    ]
    return "\n\n".join(parts) + "\n"
