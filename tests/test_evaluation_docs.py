import re
from pathlib import Path

from error_classifier.benchmark.report import read_bench
from error_classifier.config import Settings
from error_classifier.evaluation.dataset import DEFAULT_DATASET, load_dataset
from error_classifier.evaluation.metrics import DEFAULT_MODEL, summarize
from error_classifier.evaluation.results import (
    SAME_RUNTIME,
    ResultDoc,
    dataset_sha256,
    model_run,
    prompt_sha256,
    read_result,
    render_report,
)
from error_classifier.recommend import DEFAULT_TIER_GIB, choose_per_tier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = sorted((ROOT / "eval" / "results").glob("*.json"))
MARKER_START = "<!-- eval-table:start -->"
MARKER_END = "<!-- eval-table:end -->"
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
    docs = _docs()
    items = load_dataset(ROOT / DEFAULT_DATASET)
    benches = [read_bench(path) for path in sorted((ROOT / "bench" / "results").glob("*.jsonl"))]
    choices = choose_per_tier([model_run(doc, items) for doc in docs], benches)
    model = next(c.model for c in choices if c.tier_gib == DEFAULT_TIER_GIB)
    return next(doc for doc in docs if doc.meta.model == model)


def _env_example() -> dict[str, str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", text, re.MULTILINE))


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


def test_docs_tables_match_report() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count(MARKER_START) == 1
    assert text.count(MARKER_END) == 1
    between = text.split(MARKER_START)[1].split(MARKER_END)[0]
    items = load_dataset(ROOT / DEFAULT_DATASET)
    assert between.strip() == render_report(_docs(), items).strip()


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
