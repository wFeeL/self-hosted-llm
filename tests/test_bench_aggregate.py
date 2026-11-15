from collections.abc import Mapping

from error_classifier.benchmark.aggregate import (
    aggregate,
    latency,
    phase_cpu,
    speed,
)
from error_classifier.benchmark.records import (
    CgroupReading,
    Phase,
    Reading,
    RequestRecord,
    parse_cpu_stat,
    parse_int,
    parse_memory_stat,
    parse_metrics,
    quant_of,
)

METRICS_TEXT = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1234

llamacpp:tokens_predicted_seconds_total 12.5
llamacpp:requests_processing 0
llamacpp:labelled{slot="0"} 7
llamacpp:broken abc
llamacpp:nan nan
llamacpp:inf inf
single
"""


def rec(
    phase: Phase = "series",
    index: int = 0,
    outcome: str = "ok",
    wall_ms: int = 100,
    prompt: int | None = None,
    cached: int | None = None,
    completion: int | None = None,
    client: int = 0,
) -> RequestRecord:
    return RequestRecord(
        phase=phase,
        index=index,
        client=client,
        item_id=f"i{index}",
        started_s=float(index),
        status=200 if outcome == "ok" else None,
        outcome=outcome,
        wall_ms=wall_ms,
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=completion,
    )


def reading(
    phase: Phase,
    edge: str,
    at_s: float,
    metrics: Mapping[str, float] | None = None,
    llm_cpu: int | None = None,
    api_cpu: int | None = None,
    llm_anon: int | None = None,
) -> Reading:
    assert edge in ("start", "end")
    return Reading(
        phase=phase,
        edge=edge,  # pyright: ignore[reportArgumentType]
        at_s=at_s,
        metrics=dict(metrics or {}),
        cgroups=[
            CgroupReading(
                container="llm", cpu_usage_usec=llm_cpu, memory_current=None, memory_anon=llm_anon
            ),
            CgroupReading(
                container="api", cpu_usage_usec=api_cpu, memory_current=None, memory_anon=None
            ),
        ],
    )


def test_parse_metrics_keeps_names_with_labels_and_skips_noise() -> None:
    assert parse_metrics(METRICS_TEXT) == {
        "llamacpp:prompt_tokens_total": 1234.0,
        "llamacpp:tokens_predicted_seconds_total": 12.5,
        "llamacpp:requests_processing": 0.0,
        'llamacpp:labelled{slot="0"}': 7.0,
    }


def test_parse_metrics_empty() -> None:
    assert parse_metrics("") == {}


def test_parse_cpu_stat() -> None:
    assert parse_cpu_stat("usage_usec 123\nuser_usec 100\n") == 123
    assert parse_cpu_stat("user_usec 100\n") is None
    assert parse_cpu_stat("usage_usec x\n") is None


def test_parse_memory_stat_and_parse_int() -> None:
    assert parse_memory_stat("anon 100\nfile 200\nweird x\n") == {"anon": 100, "file": 200}
    assert parse_int("878600192\n") == 878600192
    assert parse_int("max\n") is None
    assert parse_int("") is None


def test_quant_of() -> None:
    assert quant_of("granite-4.1-3b-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_of("Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_of("Qwen_Qwen3.5-2B-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_of("m-Q8_0.gguf") == "Q8_0"
    assert quant_of("m-BF16.gguf") == "BF16"
    assert quant_of("model.gguf") is None


def test_latency_percentiles_use_only_ok_rows() -> None:
    rows = [
        rec(index=0, wall_ms=100),
        rec(index=1, wall_ms=200),
        rec(index=2, wall_ms=300),
        rec(index=3, outcome="http_502", wall_ms=9000),
        rec(index=4, outcome="queue_full", wall_ms=50),
    ]
    stats = latency(rows)
    assert (stats.n_ok, stats.p50_ms, stats.p95_ms, stats.max_ms) == (3, 200, 300, 300)
    assert stats.outcomes == {"http_502": 1, "ok": 3, "queue_full": 1}
    assert list(stats.outcomes) == ["http_502", "ok", "queue_full"]


def test_latency_without_ok_rows() -> None:
    stats = latency([rec(outcome="client_timeout")])
    assert stats.n_ok == 0
    assert (stats.p50_ms, stats.p95_ms, stats.max_ms) == (None, None, None)
    assert stats.outcomes == {"client_timeout": 1}
    assert latency([]).n_ok == 0


def test_speed_from_counter_deltas() -> None:
    start = reading(
        "series",
        "start",
        0,
        {
            "llamacpp:tokens_predicted_total": 100,
            "llamacpp:tokens_predicted_seconds_total": 10,
            "llamacpp:prompt_tokens_total": 1000,
            "llamacpp:prompt_seconds_total": 5,
        },
    )
    end = reading(
        "series",
        "end",
        60,
        {
            "llamacpp:tokens_predicted_total": 700,
            "llamacpp:tokens_predicted_seconds_total": 40,
            "llamacpp:prompt_tokens_total": 4000,
            "llamacpp:prompt_seconds_total": 15,
        },
    )
    stats = speed(start, end)
    assert stats.generation_tps == 20.0
    assert stats.prompt_tps == 300.0


def test_speed_missing_key_or_zero_denominator() -> None:
    start = reading("series", "start", 0, {"llamacpp:tokens_predicted_total": 1})
    end = reading("series", "end", 1, {"llamacpp:tokens_predicted_total": 2})
    assert speed(start, end).generation_tps is None
    same = {"llamacpp:tokens_predicted_total": 1, "llamacpp:tokens_predicted_seconds_total": 3}
    flat = speed(reading("series", "start", 0, same), reading("series", "end", 1, same))
    assert flat.generation_tps is None
    assert flat.prompt_tps is None


def test_phase_cpu() -> None:
    start = reading("series", "start", 10, llm_cpu=1_000_000, api_cpu=0)
    end = reading("series", "end", 12, llm_cpu=4_000_000, api_cpu=500_000)
    cpu = phase_cpu(start, end)
    assert cpu.llm_percent == 150.0
    assert cpu.api_percent == 25.0


def test_phase_cpu_missing_reading() -> None:
    start = reading("series", "start", 0, llm_cpu=None)
    end = reading("series", "end", 2, llm_cpu=5)
    assert phase_cpu(start, end).llm_percent is None
    assert phase_cpu(reading("series", "start", 1), reading("series", "end", 1)).llm_percent is None


def test_aggregate_without_sequential_phase() -> None:
    records = [rec(index=i, wall_ms=100 + i, prompt=10, cached=2, completion=5) for i in range(20)]
    records.append(rec(index=20, outcome="http_502", prompt=None))
    agg = aggregate(records, [])
    assert agg.series.n_ok == 20
    assert agg.sequential is None
    assert agg.sequential_all is None
    assert agg.long_input.n_ok == 0
    assert agg.parallel.n_ok == 0
    assert agg.queue.n == 0
    assert agg.cpu == {}
    assert agg.llm_anon_max is None
    assert agg.tokens == {"prompt": 10, "cached": 2, "completion": 5}


def test_aggregate_tokens_without_ok_rows() -> None:
    agg = aggregate([rec(outcome="http_502")], [])
    assert agg.tokens == {"prompt": None, "cached": None, "completion": None}


def test_aggregate_tokens_are_means_of_ok_rows_with_values() -> None:
    records = [
        rec(index=0, prompt=10, cached=0, completion=4),
        rec(index=1, prompt=13, cached=1, completion=None),
        rec(index=2, prompt=None),
    ]
    assert aggregate(records, []).tokens == {"prompt": 12, "cached": 0, "completion": 4}


def test_aggregate_sequential_drift_by_index() -> None:
    # Rows are given shuffled: first/last 10 follow `index`, not list order.
    records = [rec("sequential", i, wall_ms=1000 + i) for i in range(50)]
    records[49] = rec("sequential", 49, outcome="http_502", wall_ms=9)
    records.reverse()
    readings = [
        reading("sequential", "start", 0, llm_anon=1000),
        reading("sequential", "end", 100, llm_anon=1500),
    ]
    agg = aggregate(records, readings)
    assert agg.sequential is not None
    assert agg.sequential.first10.p50_ms == 1004
    assert agg.sequential.first10.n_ok == 10
    assert agg.sequential.last10.n_ok == 9
    assert agg.sequential.last10.p95_ms == 1048
    assert agg.sequential.last10.outcomes == {"http_502": 1, "ok": 9}
    assert (agg.sequential.memory_before, agg.sequential.memory_after) == (1000, 1500)
    assert agg.sequential_all is not None
    assert agg.sequential_all.n_ok == 49
    assert agg.llm_anon_max == 1500


def test_aggregate_queue_immediate_and_timeout_429() -> None:
    records = [
        rec("queue", 0, wall_ms=400),
        rec("queue", 1, outcome="queue_full", wall_ms=5),
        rec("queue", 2, outcome="queue_full", wall_ms=999),
        rec("queue", 3, outcome="queue_full", wall_ms=1000),
        rec("queue", 4, outcome="queue_full", wall_ms=30000),
        rec("queue", 5, outcome="http_429", wall_ms=3),
    ]
    q = aggregate(records, []).queue
    assert q.n == 6
    assert q.immediate_429 == 2
    assert q.timeout_429 == 2
    assert q.outcomes == {"http_429": 1, "ok": 1, "queue_full": 4}


def test_aggregate_speed_cpu_and_phases() -> None:
    counters0 = {
        "llamacpp:tokens_predicted_total": 0,
        "llamacpp:tokens_predicted_seconds_total": 0,
    }
    counters1 = {
        "llamacpp:tokens_predicted_total": 50,
        "llamacpp:tokens_predicted_seconds_total": 5,
    }
    readings = [
        reading("series", "start", 0, counters0, llm_cpu=0, api_cpu=0),
        reading("series", "end", 10, counters1, llm_cpu=10_000_000, api_cpu=1_000_000),
        reading("long", "start", 10, llm_cpu=0),
    ]
    agg = aggregate([rec("parallel", 0, wall_ms=70), rec("long", 0, wall_ms=500)], readings)
    assert agg.speed.generation_tps == 10.0
    assert agg.speed.prompt_tps is None
    assert set(agg.cpu) == {"series"}
    assert agg.cpu["series"].llm_percent == 100.0
    assert agg.cpu["series"].api_percent == 10.0
    assert agg.parallel.p95_ms == 70
    assert agg.long_input.p50_ms == 500
