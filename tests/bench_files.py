# ruff: noqa: E501
from error_classifier.benchmark.aggregate import (
    Aggregates,
    DriftStats,
    LatencyStats,
    PhaseCpu,
    QueueStats,
    SpeedStats,
)
from error_classifier.benchmark.records import RequestRecord
from error_classifier.benchmark.report import BenchConfig, BenchFile, Memory, Summary
from error_classifier.benchmark.runner import ColdStart

GIB = 2**30


def lat(
    n_ok: int, p50: int | None, p95: int | None, mx: int | None, **outcomes: int
) -> LatencyStats:
    return LatencyStats(
        n_ok=n_ok,
        outcomes={"ok": n_ok, **outcomes} if n_ok else outcomes,
        p50_ms=p50,
        p95_ms=p95,
        max_ms=mx,
    )


def config(model: str, **changes: object) -> BenchConfig:
    base = BenchConfig(
        model=model,
        engine="podman",
        compose_version="podman-compose version 1.5.0",
        compose_files=["compose.yaml", "compose.debug.yaml"],
        llm_container="llm_c",
        api_container="api_c",
        llm_image=None,
        llm_image_id=None,
        llm_image_size=None,
        api_image=None,
        api_image_id=None,
        api_image_size=None,
        runtime_env={
            "LLAMA_ARG_HF_REPO": f"repo-{model}",
            "LLAMA_ARG_HF_FILE": f"{model}.gguf",
            "LLAMA_ARG_ALIAS": model,
            "LLAMA_ARG_THREADS": "6",
            "LLAMA_ARG_CTX_SIZE": "8192",
            "LLAMA_ARG_N_PARALLEL": "2",
            "LLAMA_ARG_BATCH": "512",
        },
        api_env={"LLM_MODEL": model, "LLM_DISABLE_THINKING": "true", "MAX_QUEUE": "4"},
        quant=None,
        model_path=None,
        model_file_size=None,
        inputs_sha256="abc",
        phases={"series": 20},
        skip_sequential=False,
        cold_start=False,
        git_sha=None,
        git_dirty=None,
        started_at="2026-11-15T10:00:00Z",
        loadavg_before=None,
        loadavg_after=None,
        loaded=False,
        hawk_containers=None,
        cpu_count=8,
        machine="x86_64",
        complete=True,
        stop_reason=None,
    )
    return base.model_copy(update=changes)


def aggregates(**changes: object) -> Aggregates:
    base = Aggregates(
        series=lat(20, 5000, 5500, 6000),
        speed=SpeedStats(generation_tps=8.0, prompt_tps=90.0),
        long_input=lat(1, 9000, 9000, 9000),
        sequential=None,
        sequential_all=None,
        parallel=lat(20, 6000, 7000, 7500),
        queue=QueueStats(n=7, immediate_429=1, timeout_429=0, outcomes={"ok": 6, "queue_full": 1}),
        cpu={"series": PhaseCpu(llm_percent=400.0, api_percent=10.0)},
        tokens={"prompt": 1, "cached": 0, "completion": 1},
        llm_anon_max=2 * GIB,
    )
    return base.model_copy(update=changes)


def rows(phases: dict[str, int]) -> list[RequestRecord]:
    return [
        RequestRecord(
            phase=phase,  # pyright: ignore[reportArgumentType]
            index=i,
            client=0,
            item_id=f"id{i}",
            started_s=float(i),
            status=200,
            outcome="ok",
            wall_ms=1000,
            prompt_tokens=None,
            cached_tokens=None,
            completion_tokens=None,
        )
        for phase, n in phases.items()
        for i in range(n)
    ]


def bench_file(
    model: str,
    conf: dict[str, object] | None = None,
    agg: dict[str, object] | None = None,
    memory: Memory | None = None,
    cold_start: ColdStart | None = None,
) -> BenchFile:
    summary = Summary(
        config=config(model, **(conf or {})),
        cold_start=cold_start,
        memory=memory
        or Memory(
            llm_peak=3 * GIB,
            llm_anon=GIB,
            llm_file=GIB,
            api_peak=None,
            api_file=None,
            api_anon=None,
        ),
        aggregates=aggregates(**(agg or {})),
    )
    return BenchFile(records=rows(summary.config.phases), readings=[], summary=summary)


def alpha() -> BenchFile:
    return bench_file(
        "alpha",
        conf={
            "quant": "Q4_K_M",
            "model_file_size": 2 * GIB,
            "loadavg_before": [0.5, 0.4, 0.3],
            "llm_image_size": 3 * GIB,
            "api_image_size": 2**29,
        },
        agg={
            "series": lat(19, 4000, 6100, 7000, http_502=1),
            "speed": SpeedStats(generation_tps=12.3, prompt_tps=150.0),
            "long_input": lat(1, 9000, 9000, 9000),
            "sequential": DriftStats(
                first10=lat(10, 4000, 5000, 5000),
                last10=lat(10, 4100, 5200, 5300),
                memory_before=GIB,
                memory_after=GIB + GIB // 2,
            ),
            "sequential_all": lat(49, 4050, 5100, 5300, http_502=1),
            "parallel": lat(19, 7000, 8000, 8000, queue_full=1),
            "queue": QueueStats(
                n=7,
                immediate_429=2,
                timeout_429=1,
                outcomes={"ok": 3, "queue_full": 3, "runtime_busy": 1},
            ),
            "cpu": {"series": PhaseCpu(llm_percent=560.0, api_percent=1.0)},
            "llm_anon_max": 3 * GIB,
        },
        memory=Memory(
            llm_peak=4 * GIB,
            llm_anon=2 * GIB,
            llm_file=2**29,
            api_peak=2**28,
            api_file=2**27,
            api_anon=None,
        ),
        cold_start=ColdStart(health_s=12.04, ready_s=22.0),
    )


def beta() -> BenchFile:
    return bench_file(
        "beta",
        conf={
            "skip_sequential": True,
            "loadavg_before": [2.5, 2.0, 1.0],
            "loaded": True,
            "api_image_size": 2**29,
        },
        agg={
            "long_input": lat(0, None, None, None, client_timeout=1),
            "parallel": lat(20, 6000, 7000, 7500),
            "queue": QueueStats(n=7, immediate_429=1, timeout_429=0, outcomes={}),
        },
        memory=Memory(
            llm_peak=3 * GIB,
            llm_anon=GIB,
            llm_file=GIB,
            api_peak=None,
            api_file=None,
            api_anon=None,
        ),
    )


EXPECTED = """\
Engine: podman, podman-compose version 1.5.0; 8 vCPU x86_64; llm threads 6, context 8192, slots 2, batch 512.

| Model | Quant | GGUF | Load at start | Cold start to /health, s | Cold start to /ready, s | p50, s | p95, s | Max, s | n ok | Non-ok | Generation, tok/s | Prompt, tok/s | CPU llm, % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha | Q4_K_M | 2.00 GiB | 0.50 | 12.0 | 22.0 | 4.0 | 6.1 | 7.0 | 19 | 1 | 12.3 | 150.0 | 560.0 |
| beta | - | - | 2.50 loaded | - | - | 5.0 | 5.5 | 6.0 | 20 | 0 | 8.0 | 90.0 | 400.0 |

| Model | Long input, s | Long input outcome | Two clients p95, s | Two clients non-ok | Queue probe: immediate 429 | Queue probe: 429 after wait | Queue probe: other non-ok |
|---|---|---|---|---|---|---|---|
| alpha | 9.0 | ok | 8.0 | 1 | 2 | 1 | 1 |
| beta | - | client_timeout | 7.0 | 0 | 1 | 0 | 0 |

| Model | p50 first 10, s | p95 first 10, s | p50 last 10, s | p95 last 10, s | Memory before | Memory after | n ok | Non-ok |
|---|---|---|---|---|---|---|---|---|
| alpha | 4.0 | 5.0 | 4.1 | 5.2 | 1.00 GiB | 1.50 GiB | 49 | 1 |
| beta | - | - | - | - | - | - | - | - |

| Model | llm peak | llm file | api peak | api file | GGUF | RAM floor | RAM, conservative | RAM, worst | llm image | api image |
|---|---|---|---|---|---|---|---|---|---|---|
| alpha | 4.00 GiB | 0.50 GiB | 0.25 GiB | 0.12 GiB | 2.00 GiB | 3.00 GiB | 5.00 GiB | 6.00 GiB | 3.00 GiB | 0.50 GiB |
| beta | 3.00 GiB | 1.00 GiB | - | - | - | 2.00 GiB | - | - | - | 0.50 GiB |

| Model | llm need | api peak | Disk |
|---|---|---|---|
| alpha | 5.00 GiB | 0.25 GiB | 5.50 GiB |
| beta | - | - | - |
"""
