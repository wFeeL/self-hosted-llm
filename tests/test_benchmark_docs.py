import re
from pathlib import Path

from error_classifier.benchmark.aggregate import aggregate
from error_classifier.benchmark.report import BenchFile, read_bench, render_bench_report

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "bench" / "results").glob("*.jsonl"))
M2_FILES = sorted((ROOT / "bench" / "results-m2").glob("*.jsonl"))
DOC = ROOT / "README.md"
MARKER_START = "<!-- bench-table:start -->"
MARKER_END = "<!-- bench-table:end -->"
M2_START = "<!-- bench-m2-table:start -->"
M2_END = "<!-- bench-m2-table:end -->"
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


def test_tables_in_the_doc_match_the_report() -> None:
    text = DOC.read_text(encoding="utf-8")
    start, end = text.index(MARKER_START), text.index(MARKER_END)
    block = text[start + len(MARKER_START) : end].strip()

    assert block == render_bench_report(_benches()).strip()


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


def test_m2_tables_in_the_doc_match_the_report() -> None:
    text = DOC.read_text(encoding="utf-8")
    start, end = text.index(M2_START), text.index(M2_END)
    block = text[start + len(M2_START) : end].strip()

    assert block == render_bench_report(_m2_benches()).strip()
