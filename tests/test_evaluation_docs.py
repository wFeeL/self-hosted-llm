import json
import re
from pathlib import Path

from docs_helpers import marked_block, readme, report_cell, report_tables, table_rows, tier_choice

from error_classifier.benchmark.report import read_bench, render_bench_report
from error_classifier.config import Settings
from error_classifier.evaluation.dataset import DEFAULT_DATASET, load_dataset
from error_classifier.evaluation.metrics import DEFAULT_MODEL, summarize
from error_classifier.evaluation.results import (
    SAME_RUNTIME,
    ResultDoc,
    dataset_sha256,
    prompt_sha256,
    read_result,
)
from error_classifier.recommend import DEFAULT_TIER_GIB

ROOT = Path(__file__).resolve().parents[1]
RESULTS = sorted((ROOT / "eval" / "results").glob("*.json"))
REQUIRED_RUNTIME = (
    "LLAMA_ARG_CTX_SIZE",
    "LLAMA_ARG_N_PARALLEL",
    "LLAMA_ARG_THREADS",
    "LLAMA_ARG_BATCH",
)
EXPECTED_MODELS = {
    "granite-4.0-1b-q4km",
    "granite-4.1-3b-q4km",
    "qwen3-4b-instruct-2507-q4km",
    "qwen3.5-2b-q4km",
}
THINKING_MODELS = {"qwen3.5-2b-q4km"}


def _docs() -> list[ResultDoc]:
    return [read_result(path) for path in RESULTS]


def _default(field: str) -> object:
    return Settings.model_fields[field].default


def _default_tier_doc() -> ResultDoc:
    return tier_choice(DEFAULT_TIER_GIB)[1]


def _env_example() -> dict[str, str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Z_][A-Z0-9_]*)=(.*)$", text, re.MULTILINE))


def test_four_results_exist() -> None:
    docs = _docs()
    assert {doc.meta.model for doc in docs} == EXPECTED_MODELS
    assert len(RESULTS) == 4
    for path, doc in zip(RESULTS, docs, strict=True):
        assert path.name == f"{doc.meta.model}.json"


def test_results_match_this_set_and_prompt() -> None:
    dataset = dataset_sha256(ROOT / DEFAULT_DATASET)
    prompt = prompt_sha256()
    for doc in _docs():
        assert doc.meta.complete, doc.meta.model
        assert doc.summary.items == 60, doc.meta.model
        assert len(doc.items) == 60, doc.meta.model
        assert doc.meta.dataset_sha256 == dataset, doc.meta.model
        assert doc.meta.prompt_sha256 == prompt, doc.meta.model


def test_results_share_settings() -> None:
    docs = _docs()
    defaults = {
        "temperature": _default("llm_temperature"),
        "max_output_tokens": _default("max_output_tokens"),
        "max_input_chars": _default("max_input_chars"),
        "request_timeout_s": _default("request_timeout_s"),
    }
    for doc in docs:
        shared = doc.meta.settings.model_dump(exclude={"disable_thinking"})
        assert shared == defaults, doc.meta.model
        assert doc.meta.settings.disable_thinking == (doc.meta.model in THINKING_MODELS)
        assert doc.meta.runtime_env["LLAMA_ARG_ALIAS"] == doc.meta.model
        assert doc.summary.truncated == 0, doc.meta.model
        for key in REQUIRED_RUNTIME:
            assert key in doc.meta.runtime_env, (doc.meta.model, key)
        for key in SAME_RUNTIME:
            assert doc.meta.runtime_env.get(key) == docs[0].meta.runtime_env.get(key), key


def test_stored_summary_matches_items() -> None:
    items = load_dataset(ROOT / DEFAULT_DATASET)
    for doc in _docs():
        assert summarize(items, doc.items) == doc.summary, doc.meta.model


def test_model_table_matches_the_results() -> None:
    items = load_dataset(ROOT / DEFAULT_DATASET)
    docs = {doc.meta.model: doc for doc in _docs()}
    bench_files = [
        read_bench(path) for path in sorted((ROOT / "bench" / "results").glob("*.jsonl"))
    ]
    measured = {b.summary.config.model for b in bench_files}
    report = render_bench_report(bench_files)
    rows = table_rows(marked_block(readme(), "model-table"))
    assert {row[0].strip("`") for row in rows} == set(docs)
    assert len(rows) == len(docs)
    for row in rows:
        alias = row[0].strip("`")
        summary = summarize(items, docs[alias].items)
        both = re.fullmatch(r"(\d+)( из (\d+))?", row[2])
        assert both is not None, (alias, row[2])
        assert int(both.group(1)) == summary.both_correct, alias
        if both.group(3) is not None:
            assert int(both.group(3)) == len(items), alias
        assert row[3] == str(summary.category_correct), alias
        assert row[4] == str(summary.severity_correct), alias
        assert row[5] == f"{summary.leaks} из {summary.secret_ok_items}", alias
        if alias in measured:
            tables = report_tables(report, alias)
            assert row[6] == report_cell(tables, "p50, s"), alias
            assert row[7] == report_cell(tables, "llm need").replace("GiB", "ГиБ"), alias
        else:
            assert row[6:8] == ["-", "-"], alias


def test_readme_example_is_the_js_001_answer() -> None:
    text = readme()
    model = _env_example()["MODEL_ALIAS"]
    doc = next(d for d in _docs() if d.meta.model == model)
    row = next(r for r in doc.items if r.id == "js-001")
    lines = [line for line in text.splitlines() if line.startswith('{"category"')]
    assert len(lines) == 1
    body = json.loads(lines[0])
    assert body["category"] == str(row.category)
    assert body["severity"] == str(row.severity)
    assert body["summary"] == row.summary
    assert body["suggested_action"] == row.suggested_action
    assert body["model"] == model
    assert body["latency_ms"] == row.latency_ms
    assert body["truncated"] is row.truncated
    requests = [line for line in text.splitlines() if line.startswith('{"message"')]
    assert len(requests) == 1
    item = next(i for i in load_dataset(ROOT / DEFAULT_DATASET) if i.id == "js-001")
    assert json.loads(requests[0])["message"] == item.text


def test_env_example_model_is_the_recommendation_for_the_default_tier() -> None:
    assert _env_example()["MODEL_ALIAS"] == _default_tier_doc().meta.model


def test_env_example_thinking_matches_the_recommended_run() -> None:
    flag = _env_example()["LLM_DISABLE_THINKING"].lower() == "true"
    assert flag == _default_tier_doc().meta.settings.disable_thinking


def test_env_example_model_files_match_the_recommended_run() -> None:
    env = _env_example()
    meta = _default_tier_doc().meta
    for key in ("LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_FILE"):
        assert env[key] == meta.runtime_env[key], key


def test_default_model_is_the_env_example_model() -> None:
    assert _env_example()["MODEL_ALIAS"] == DEFAULT_MODEL
