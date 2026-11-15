from collections import Counter
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from error_classifier.benchmark.records import Phase, Reading, RequestRecord
from error_classifier.evaluation.metrics import nearest_rank

IMMEDIATE_429_MS: Final = 1000
DRIFT_ROWS: Final = 10

_PREDICTED: Final = "llamacpp:tokens_predicted_total"
_PREDICTED_S: Final = "llamacpp:tokens_predicted_seconds_total"
_PROMPT: Final = "llamacpp:prompt_tokens_total"
_PROMPT_S: Final = "llamacpp:prompt_seconds_total"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LatencyStats(_Model):
    n_ok: int
    outcomes: dict[str, int]  # by outcome, keys sorted
    p50_ms: int | None
    p95_ms: int | None
    max_ms: int | None


class SpeedStats(_Model):
    generation_tps: float | None
    prompt_tps: float | None


class PhaseCpu(_Model):
    llm_percent: float | None
    api_percent: float | None


class DriftStats(_Model):
    first10: LatencyStats
    last10: LatencyStats
    memory_before: int | None  # llm memory_anon at the start of the phase
    memory_after: int | None  # llm memory_anon at the end


class QueueStats(_Model):
    n: int
    immediate_429: int  # outcome queue_full with wall_ms < 1000
    timeout_429: int
    outcomes: dict[str, int]


class Aggregates(_Model):
    series: LatencyStats
    speed: SpeedStats
    long_input: LatencyStats
    sequential: DriftStats | None  # None when skipped
    sequential_all: LatencyStats | None
    parallel: LatencyStats
    queue: QueueStats
    cpu: dict[str, PhaseCpu]  # by phase present in readings
    tokens: dict[str, int | None]  # keys "prompt", "cached", "completion"
    llm_anon_max: int | None


def _outcomes(records: Sequence[RequestRecord]) -> dict[str, int]:
    return dict(sorted(Counter(r.outcome for r in records).items()))


def latency(records: Sequence[RequestRecord]) -> LatencyStats:
    ok = [r.wall_ms for r in records if r.outcome == "ok"]
    return LatencyStats(
        n_ok=len(ok),
        outcomes=_outcomes(records),
        p50_ms=nearest_rank(ok, 50) if ok else None,
        p95_ms=nearest_rank(ok, 95) if ok else None,
        max_ms=max(ok) if ok else None,
    )


def _rate(start: Reading, end: Reading, amount: str, seconds: str) -> float | None:
    keys = (amount, seconds)
    if any(k not in start.metrics or k not in end.metrics for k in keys):
        return None
    elapsed = end.metrics[seconds] - start.metrics[seconds]
    if elapsed == 0:
        return None
    return round((end.metrics[amount] - start.metrics[amount]) / elapsed, 1)


def speed(start: Reading, end: Reading) -> SpeedStats:
    return SpeedStats(
        generation_tps=_rate(start, end, _PREDICTED, _PREDICTED_S),
        prompt_tps=_rate(start, end, _PROMPT, _PROMPT_S),
    )


def _cpu_usec(reading: Reading, container: str) -> int | None:
    for cg in reading.cgroups:
        if cg.container == container:
            return cg.cpu_usage_usec
    return None


def _cpu_percent(start: Reading, end: Reading, container: str) -> float | None:
    before, after = _cpu_usec(start, container), _cpu_usec(end, container)
    wall = end.at_s - start.at_s
    if before is None or after is None or wall <= 0:
        return None
    return round((after - before) / (wall * 1e6) * 100, 1)


def phase_cpu(start: Reading, end: Reading) -> PhaseCpu:
    return PhaseCpu(
        llm_percent=_cpu_percent(start, end, "llm"),
        api_percent=_cpu_percent(start, end, "api"),
    )


def _llm_anon(reading: Reading | None) -> int | None:
    if reading is None:
        return None
    return next((c.memory_anon for c in reading.cgroups if c.container == "llm"), None)


def _edges(readings: Sequence[Reading]) -> dict[Phase, dict[str, Reading]]:
    edges: dict[Phase, dict[str, Reading]] = {}
    for r in readings:
        edges.setdefault(r.phase, {})[r.edge] = r
    return edges


def _mean(values: list[int]) -> int | None:
    return round(sum(values) / len(values)) if values else None


def _tokens(series: Sequence[RequestRecord]) -> dict[str, int | None]:
    ok = [r for r in series if r.outcome == "ok"]

    def column(values: list[int | None]) -> int | None:
        return _mean([v for v in values if v is not None])

    return {
        "prompt": column([r.prompt_tokens for r in ok]),
        "cached": column([r.cached_tokens for r in ok]),
        "completion": column([r.completion_tokens for r in ok]),
    }


def _queue(records: Sequence[RequestRecord]) -> QueueStats:
    full = [r for r in records if r.outcome == "queue_full"]
    immediate = sum(1 for r in full if r.wall_ms < IMMEDIATE_429_MS)
    return QueueStats(
        n=len(records),
        immediate_429=immediate,
        timeout_429=len(full) - immediate,
        outcomes=_outcomes(records),
    )


def aggregate(records: Sequence[RequestRecord], readings: Sequence[Reading]) -> Aggregates:
    by_phase: dict[Phase, list[RequestRecord]] = {}
    for r in records:
        by_phase.setdefault(r.phase, []).append(r)
    for rows in by_phase.values():
        rows.sort(key=lambda r: r.index)
    edges = _edges(readings)

    sequential = by_phase.get("sequential")
    drift = None
    if sequential:
        seq_edges = edges.get("sequential", {})
        drift = DriftStats(
            first10=latency(sequential[:DRIFT_ROWS]),
            last10=latency(sequential[-DRIFT_ROWS:]),
            memory_before=_llm_anon(seq_edges.get("start")),
            memory_after=_llm_anon(seq_edges.get("end")),
        )

    series_edges = edges.get("series", {})
    start, end = series_edges.get("start"), series_edges.get("end")
    anon = [a for r in readings if (a := _llm_anon(r)) is not None]
    return Aggregates(
        series=latency(by_phase.get("series", [])),
        speed=speed(start, end)
        if start and end
        else SpeedStats(generation_tps=None, prompt_tps=None),
        long_input=latency(by_phase.get("long", [])),
        sequential=drift,
        sequential_all=latency(sequential) if sequential else None,
        parallel=latency(by_phase.get("parallel", [])),
        queue=_queue(by_phase.get("queue", [])),
        cpu={
            phase: phase_cpu(e["start"], e["end"])
            for phase, e in edges.items()
            if "start" in e and "end" in e
        },
        tokens=_tokens(by_phase.get("series", [])),
        llm_anon_max=max(anon) if anon else None,
    )
