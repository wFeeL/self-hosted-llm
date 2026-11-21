import re
from pathlib import Path

from docs_helpers import marked_block, report_cell, report_tables, table_rows

from error_classifier.benchmark.aggregate import aggregate
from error_classifier.benchmark.report import BenchFile, read_bench, render_bench_report

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "bench" / "results").glob("*.jsonl"))
M2_FILES = sorted((ROOT / "bench" / "results-m2").glob("*.jsonl"))
DOC = ROOT / "README.md"
# The 50-request series was run once, before the default moved to a model without it.
SERIES_MODEL = "qwen3.5-2b-q4km"
EXPECTED_MODELS = {"granite-4.1-3b-q4km", "qwen3-4b-instruct-2507-q4km", "qwen3.5-2b-q4km"}


def _benches() -> list[BenchFile]:
    return [read_bench(path) for path in FILES]


def _m2_benches() -> list[BenchFile]:
    return [read_bench(path) for path in M2_FILES]


def _default_alias() -> str:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    match = re.search(r"^MODEL_ALIAS=(\S+)$", text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_three_complete_files_for_the_three_models() -> None:
    benches = _benches()

    assert sorted(b.summary.config.model for b in benches) == sorted(EXPECTED_MODELS)
    for path, bench in zip(FILES, benches, strict=True):
        config = bench.summary.config
        assert config.complete, path.name
        assert config.stop_reason is None, path.name
        assert path.name == f"{config.started_at[:10]}-{config.model}.jsonl"


def test_summary_aggregates_are_recomputed_from_the_rows() -> None:
    assert FILES, "no benchmark files"
    for path, bench in zip(FILES, _benches(), strict=True):
        assert aggregate(bench.records, bench.readings) == bench.summary.aggregates, path.name


def test_only_one_model_ran_the_sequential_series() -> None:
    models = {b.summary.config.model: b.summary.config for b in _benches()}

    assert SERIES_MODEL in models
    assert models[SERIES_MODEL].skip_sequential is False
    for model, config in models.items():
        if model != SERIES_MODEL:
            assert config.skip_sequential is True, model


def test_todo_names_the_missing_series() -> None:
    default = _default_alias()
    config = next(b.summary.config for b in _benches() if b.summary.config.model == default)
    if not config.skip_sequential:
        return
    readme = DOC.read_text(encoding="utf-8")
    assert "## \u041e тестовом задании" in readme
    section = " ".join(readme.split("## \u041e тестовом задании", 1)[1].split())
    assert "серию из 50 запросов" in section
    assert default in section


def test_every_run_probed_the_queue_and_kept_the_series_clean() -> None:
    for path, bench in zip(FILES, _benches(), strict=True):
        aggregates = bench.summary.aggregates
        assert aggregates.queue.immediate_429 >= 1, path.name
        assert set(aggregates.series.outcomes) == {"ok"}, path.name


def test_m2_file_is_one_complete_docker_run() -> None:
    assert len(M2_FILES) == 1, [path.name for path in M2_FILES]
    path, bench = M2_FILES[0], _m2_benches()[0]
    config = bench.summary.config

    assert config.complete, path.name
    assert config.stop_reason is None, path.name
    assert config.engine == "docker", path.name
    assert config.machine == "arm64", path.name
    assert config.git_dirty is False, path.name
    assert path.name == f"{config.started_at[:10]}-{config.model}.jsonl"
    assert aggregate(bench.records, bench.readings) == bench.summary.aggregates, path.name


COLUMNS = {
    "20 запросов подряд, p50 и p95": ["p50, s", "p95, s"],
    "Вход в 3000 символов": ["Long input, s"],
    "Два клиента сразу, p95": ["Two clients p95, s"],
    "50 запросов подряд, p50 первых и последних 10": ["p50 first 10, s", "p50 last 10, s", "n ok"],
    "Старт до `/ready`, модель уже в томе": ["Cold start to /ready, s"],
    "Генерация": ["Generation, tok/s"],
    "CPU `llm`, 100% на ядро": ["CPU llm, %"],
    "Память `llm` на пике": ["llm peak"],
    "Память `api` на пике": ["api peak"],
    "Диск: образы и модель": ["Disk"],
}
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
NOT_RUN = "не запускалась"


def test_bench_summary_matches_the_reports() -> None:
    model = _default_alias()
    rows = table_rows(marked_block(DOC.read_text(encoding="utf-8"), "bench-summary"))
    assert {row[0] for row in rows} == set(COLUMNS)
    assert len(rows) == len(COLUMNS)
    server = report_tables(render_bench_report(_benches()), model)
    m2 = report_tables(render_bench_report(_m2_benches()), model)
    for row in rows:
        for cell, tables in ((row[1], server), (row[2], m2)):
            expected: list[str] = []
            for column in COLUMNS[row[0]]:
                anchor = COLUMNS[row[0]][0] if column == "n ok" else None
                expected.append(report_cell(tables, column, anchor=anchor))
            if all(value == "-" for value in expected):
                assert cell == NOT_RUN, row[0]
            else:
                numbers = [v for value in expected for v in NUMBER.findall(value)]
                assert NUMBER.findall(cell) == numbers, row[0]
    sequential = [r for r in _m2_benches()[0].records if r.phase == "sequential"]
    assert len(sequential) == 50
    assert {r.outcome for r in sequential} == {"ok"}
