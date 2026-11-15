import re
from pathlib import Path

from error_classifier.benchmark.aggregate import aggregate
from error_classifier.benchmark.report import BenchFile, read_bench, render_bench_report

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "bench" / "results").glob("*.jsonl"))
DOC = ROOT / "README.md"
MARKER_START = "<!-- bench-table:start -->"
MARKER_END = "<!-- bench-table:end -->"
EXPECTED_MODELS = {"granite-4.1-3b-q4km", "qwen3-4b-instruct-2507-q4km", "qwen3.5-2b-q4km"}


def _benches() -> list[BenchFile]:
    return [read_bench(path) for path in FILES]


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


def test_only_the_default_model_runs_the_sequential_series() -> None:
    default = _default_alias()
    assert default in EXPECTED_MODELS
    models = {b.summary.config.model: b.summary.config for b in _benches()}

    assert default in models
    assert models[default].skip_sequential is False
    for model, config in models.items():
        if model != default:
            assert config.skip_sequential is True, model


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
