import re

from docs_helpers import ROOT

from error_classifier.benchmark.report import read_bench
from error_classifier.selection.presets import Candidate, load_presets

README = (ROOT / "README.md").read_text(encoding="utf-8")
PRESETS = ROOT / "select" / "presets.ini"


def _readme_blocks() -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    for alias, body in re.findall(
        r"<!-- model-env:(\S+?):start -->(.*?)<!-- model-env:\1:end -->", README, re.DOTALL
    ):
        blocks[alias] = dict(re.findall(r"^([A-Z_]+)=(.*)$", body, re.MULTILINE))
    return blocks


def _candidates() -> dict[str, Candidate]:
    return {c.alias: c for c in load_presets(PRESETS)}


def test_presets_match_readme_blocks() -> None:
    blocks = _readme_blocks()
    candidates = _candidates()
    assert blocks
    assert set(candidates) == set(blocks)
    for alias, env in blocks.items():
        candidate = candidates[alias]
        assert candidate.hf_repo == env["LLAMA_ARG_HF_REPO"], alias
        assert candidate.hf_file == env["LLAMA_ARG_HF_FILE"], alias
        assert candidate.disable_thinking == (env["LLM_DISABLE_THINKING"] == "true"), alias


def test_presets_have_reference_runs() -> None:
    for alias in _candidates():
        assert (ROOT / "eval" / "results" / f"{alias}.json").is_file(), alias


def test_presets_bench_names_are_known() -> None:
    known = set(_candidates())
    for path in sorted((ROOT / "bench" / "results").glob("*.jsonl")):
        assert read_bench(path).summary.config.model in known, path.name
