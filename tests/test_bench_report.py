import dataclasses
from pathlib import Path

import pytest
from bench_files import EXPECTED, GIB, alpha, bench_file, beta, config, rows

from error_classifier.benchmark.records import CgroupReading, Reading, RequestRecord
from error_classifier.benchmark.report import (
    BenchFile,
    Memory,
    Summary,
    cpu_limit,
    read_bench,
    render_bench_report,
    write_bench,
)


def test_report_reference_for_two_files() -> None:
    # Input order is reversed on purpose: rows follow the model name.
    assert render_bench_report([beta(), alpha()]) == EXPECTED


def test_report_refuses_nothing_to_report() -> None:
    with pytest.raises(ValueError, match="no results"):
        render_bench_report([])


def test_report_refuses_incomplete_file() -> None:
    partial = bench_file("alpha", conf={"complete": False, "stop_reason": "time_limit"})
    with pytest.raises(ValueError, match=r"alpha.*incomplete"):
        render_bench_report([partial, beta()])


def test_report_refuses_different_inputs_and_engines() -> None:
    with pytest.raises(ValueError, match="inputs"):
        render_bench_report([alpha(), bench_file("beta", conf={"inputs_sha256": "other"})])
    with pytest.raises(ValueError, match="engine"):
        render_bench_report([alpha(), bench_file("beta", conf={"engine": "docker"})])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_image_id", "sha-other"),
        ("cpu_count", 4),
        ("machine", "aarch64"),
        ("compose_version", "podman-compose version 9"),
    ],
)
def test_report_refuses_different_stacks_and_machines(field: str, value: object) -> None:
    first = bench_file("alpha", conf={"llm_image_id": "sha-a"})
    second = bench_file("beta", conf={"llm_image_id": "sha-a", field: value})
    with pytest.raises(ValueError, match=field.replace("_", " ")):
        render_bench_report([first, second])


def test_report_accepts_different_api_images() -> None:
    first = bench_file("alpha", conf={"api_image_id": "sha-1"})
    second = bench_file("beta", conf={"api_image_id": "sha-2"})
    assert "| beta |" in render_bench_report([first, second])


def test_report_refuses_records_that_do_not_match_the_planned_phases() -> None:
    short = bench_file("beta")
    short = BenchFile(short.records[:-1], short.readings, short.summary)
    with pytest.raises(ValueError, match=r"beta.*series.*19.*20"):
        render_bench_report([alpha(), short])
    extra = bench_file("beta", conf={"phases": {"series": 20, "queue": 7}})
    extra = BenchFile(rows({"series": 20}), extra.readings, extra.summary)
    with pytest.raises(ValueError, match=r"beta.*queue"):
        render_bench_report([alpha(), extra])


def test_report_refuses_different_runtime_settings() -> None:
    other = config("beta").runtime_env | {"LLAMA_ARG_THREADS": "4"}
    with pytest.raises(ValueError, match="LLAMA_ARG_THREADS"):
        render_bench_report([alpha(), bench_file("beta", conf={"runtime_env": other})])


def test_report_refuses_different_api_settings() -> None:
    other = config("beta").api_env | {"MAX_QUEUE": "9"}
    with pytest.raises(ValueError, match="MAX_QUEUE"):
        render_bench_report([alpha(), bench_file("beta", conf={"api_env": other})])


def test_report_accepts_models_that_differ_in_the_allowed_keys() -> None:
    other = config("beta").api_env | {"LLM_DISABLE_THINKING": "false"}
    text = render_bench_report([alpha(), bench_file("beta", conf={"api_env": other})])
    assert "| beta |" in text


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    record = RequestRecord(
        phase="series",
        index=0,
        client=0,
        item_id="i",
        started_s=0.5,
        status=200,
        outcome="ok",
        wall_ms=12,
        prompt_tokens=1,
        cached_tokens=0,
        completion_tokens=2,
    )
    reading = Reading(
        phase="series",
        edge="start",
        at_s=0.5,
        metrics={"llamacpp:requests_processing": 0.0},
        cgroups=[CgroupReading(container="llm", cpu_usage_usec=1, memory_current=2, memory_anon=3)],
    )
    summary = alpha().summary
    path = tmp_path / "out" / "run.jsonl"

    write_bench(path, [record], [reading], summary)
    loaded = read_bench(path)

    assert loaded == BenchFile([record], [reading], summary)
    assert [p.name for p in path.parent.iterdir()] == ["run.jsonl"]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line.split('"type":"')[1].split('"')[0] for line in lines] == [
        "request",
        "reading",
        "summary",
    ]


def test_read_requires_summary_last(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_bench(path, [], [], alpha().summary)
    lines = path.read_text(encoding="utf-8").splitlines()
    request = (
        '{"type":"request","phase":"series","index":0,"client":0,"item_id":"i","started_s":0.0,'
        '"status":200,"outcome":"ok","wall_ms":1,"prompt_tokens":null,"cached_tokens":null,'
        '"completion_tokens":null}'
    )
    path.write_text("\n".join([*lines, request]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        read_bench(path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("\n", "empty"),
        ("not json\n", "Expecting"),
        ('{"type":"mystery"}\n', "unknown row type"),
        ('{"type":"summary"}\n', "validation error"),
    ],
)
def test_read_refuses_damaged_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_bench(path)


def test_partial_file_is_readable_but_not_reportable(tmp_path: Path) -> None:
    partial = bench_file("alpha", conf={"complete": False, "stop_reason": "time_limit"})
    path = tmp_path / "run.jsonl.partial"
    write_bench(path, [], [], partial.summary)
    loaded = read_bench(path)
    assert loaded.summary.config.stop_reason == "time_limit"
    with pytest.raises(ValueError, match="incomplete"):
        render_bench_report([loaded])


def test_summary_without_api_file_still_reads() -> None:
    data = alpha().summary.model_dump(mode="json")
    del data["memory"]["api_file"]
    assert Summary.model_validate(data).memory.api_file is None


def test_old_summary_without_swap_fields_reads() -> None:
    data = alpha().summary.model_dump(mode="json")
    for key in (
        "llm_swap_peak",
        "api_swap_peak",
        "llm_swap_max",
        "api_swap_max",
        "llm_memory_max",
        "api_memory_max",
    ):
        del data["memory"][key]
    memory = Summary.model_validate(data).memory
    assert memory.llm_swap_peak is None
    assert memory.api_swap_max is None
    assert memory.llm_memory_max is None


def _memory_table(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Model | llm peak |"))

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    return cells(lines[start]), cells(lines[start + 2])


def test_report_shows_swap_and_limits_after_api_file() -> None:
    memory = Memory(
        llm_peak=GIB,
        llm_anon=GIB,
        llm_file=GIB,
        api_peak=GIB,
        api_anon=None,
        api_file=GIB,
        llm_swap_peak=2**29,
        llm_swap_max="0",
        llm_memory_max="6442450944",
        api_swap_peak=2**28,
        api_swap_max="max",
        api_memory_max="536870912",
    )
    header, row = _memory_table(render_bench_report([bench_file("alpha", memory=memory)]))
    columns = dict(zip(header, row, strict=True))
    start = header.index("api file")
    assert header[start + 1 : start + 7] == [
        "llm swap peak",
        "llm swap max",
        "llm memory max",
        "api swap peak",
        "api swap max",
        "api memory max",
    ]
    assert columns["llm swap peak"] == "0.50 GiB"
    assert columns["api swap peak"] == "0.25 GiB"
    assert (columns["llm swap max"], columns["llm memory max"]) == ("0", "6442450944")
    assert (columns["api swap max"], columns["api memory max"]) == ("max", "536870912")


def test_report_shows_dashes_for_files_without_swap_fields() -> None:
    header, row = _memory_table(render_bench_report([bench_file("alpha")]))
    columns = dict(zip(header, row, strict=True))
    for name in (
        "llm swap peak",
        "llm swap max",
        "llm memory max",
        "api swap peak",
        "api swap max",
        "api memory max",
    ):
        assert columns[name] == "-", name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("400000 100000", "4"),
        ("150000 100000", "1.5"),
        ("333333 100000", "3.33"),
        ("max 100000", "none"),
        ("garbage", "garbage"),
        ("400000", "400000"),
        ("400000 0", "400000 0"),
        ("a b", "a b"),
    ],
)
def test_cpu_limit_formats(raw: str, expected: str) -> None:
    assert cpu_limit(raw) == expected


def with_cpu_limit(bench: BenchFile, value: str | None) -> BenchFile:
    config = bench.summary.config.model_copy(update={"llm_cpu_max": value})
    summary = bench.summary.model_copy(update={"config": config})
    return dataclasses.replace(bench, summary=summary)


def test_report_shows_the_cpu_limit() -> None:
    files = [with_cpu_limit(alpha(), "400000 100000"), with_cpu_limit(beta(), "400000 100000")]
    assert "llm CPU limit 4;" in render_bench_report(files)


def test_report_refuses_mixed_cpu_limits() -> None:
    files = [with_cpu_limit(alpha(), "400000 100000"), beta()]
    with pytest.raises(ValueError, match="llm CPU limits"):
        render_bench_report(files)


def started_at(bench: BenchFile, value: str) -> BenchFile:
    config = bench.summary.config.model_copy(update={"started_at": value})
    summary = bench.summary.model_copy(update={"config": config})
    return dataclasses.replace(bench, summary=summary)


def test_report_refuses_two_files_of_one_model() -> None:
    first = started_at(alpha(), "2026-11-15T10:00:00Z")
    second = started_at(alpha(), "2026-11-16T11:30:00Z")
    with pytest.raises(ValueError, match="two result files") as refused:
        render_bench_report([first, second])
    text = str(refused.value)
    assert "alpha" in text
    assert "2026-11-15T10:00:00Z" in text
    assert "2026-11-16T11:30:00Z" in text
