import re

from docs_helpers import ROOT, marked_block, readme, tier_choice

from error_classifier.benchmark.report import read_bench
from error_classifier.evaluation.results import read_result
from error_classifier.selection.presets import Candidate, load_presets

PRESETS = ROOT / "select" / "presets.ini"


def _candidates() -> dict[str, Candidate]:
    return {c.alias: c for c in load_presets(PRESETS)}


def test_presets_match_the_evaluation_runs() -> None:
    runs = {
        doc.meta.model: doc.meta
        for doc in map(read_result, sorted((ROOT / "eval" / "results").glob("*.json")))
    }
    candidates = _candidates()
    assert runs
    assert set(candidates) == set(runs)
    for alias, meta in runs.items():
        candidate = candidates[alias]
        assert candidate.hf_repo == meta.runtime_env["LLAMA_ARG_HF_REPO"], alias
        assert candidate.hf_file == meta.runtime_env["LLAMA_ARG_HF_FILE"], alias
        assert candidate.disable_thinking == meta.settings.disable_thinking, alias


def test_presets_have_reference_runs() -> None:
    for alias in _candidates():
        assert (ROOT / "eval" / "results" / f"{alias}.json").is_file(), alias


def test_presets_bench_names_are_known() -> None:
    known = set(_candidates())
    for path in sorted((ROOT / "bench" / "results").glob("*.jsonl")):
        assert read_bench(path).summary.config.model in known, path.name


def test_small_model_block_matches_the_preset() -> None:
    block = marked_block(readme(), "small-model-env")
    env = dict(re.findall(r"^([A-Z_]+)=(.*)$", block, re.MULTILINE))
    candidates = _candidates()
    alias = env["MODEL_ALIAS"]
    assert alias in candidates
    preset = candidates[alias]
    assert preset.hf_repo == env["LLAMA_ARG_HF_REPO"]
    assert preset.hf_file == env["LLAMA_ARG_HF_FILE"]
    assert preset.disable_thinking == (env["LLM_DISABLE_THINKING"] == "true")
    model, doc = tier_choice(4)
    assert alias == model
    assert doc.meta.settings.disable_thinking == preset.disable_thinking
