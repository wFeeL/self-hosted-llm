import json
import re

import pytest
from docs_helpers import ROOT, env_example

from error_classifier.benchmark.report import BenchFile, llm_need_bytes, read_bench
from error_classifier.evaluation.results import read_result
from error_classifier.recommend import recommend_text

README = (ROOT / "README.md").read_text(encoding="utf-8")
RESULTS = {
    doc.meta.model: doc for doc in map(read_result, sorted((ROOT / "eval/results").glob("*.json")))
}
BENCHES = {
    bench.summary.config.model: bench
    for bench in map(read_bench, sorted((ROOT / "bench/results").glob("*.jsonl")))
}
REQUIRED = (
    "MODEL_ALIAS",
    "LLAMA_ARG_HF_REPO",
    "LLAMA_ARG_HF_FILE",
    "LLM_DISABLE_THINKING",
)
NOT_MEASURED = "# Memory not measured"
RECOMMENDED = re.compile(r"^Recommended: \x60([^\x60]+)\x60$", re.MULTILINE)
LIMIT = re.compile(r"^(\d+)([gm])$")
LONG_INPUT_MARGIN = 1.3
# The peak was taken under the limit it is compared with, so it understates the need.
MEMORY_MARGIN = 1.1


def _between(text: str, start: str, end: str) -> str:
    assert text.count(start) == 1, start
    assert text.count(end) == 1, end
    return text.split(start)[1].split(end)[0]


def _block(alias: str) -> str:
    return _between(README, f"<!-- model-env:{alias}:start -->", f"<!-- model-env:{alias}:end -->")


def _pairs(alias: str) -> dict[str, str]:
    return dict(re.findall(r"^([A-Z_]+)=(.*)$", _block(alias), re.MULTILINE))


def _limit_bytes(value: str) -> int:
    match = LIMIT.fullmatch(value)
    assert match is not None, value
    return int(match.group(1)) * (2**30 if match.group(2) == "g" else 2**20)


def _long_record_outcome(bench: BenchFile) -> str:
    outcomes = [r.outcome for r in bench.records if r.phase == "long"]
    assert len(outcomes) == 1
    return outcomes[0]


def test_every_evaluated_model_has_a_block() -> None:
    found = re.findall(r"<!-- model-env:(\S+?):start -->", README)
    assert len(found) == len(set(found))
    assert set(found) == set(RESULTS)


@pytest.mark.parametrize("alias", sorted(RESULTS))
def test_block_matches_the_evaluation_run(alias: str) -> None:
    pairs = _pairs(alias)
    meta = RESULTS[alias].meta
    for key in REQUIRED:
        assert key in pairs, key
    assert pairs["MODEL_ALIAS"] == alias
    assert pairs["LLAMA_ARG_HF_REPO"] == meta.runtime_env["LLAMA_ARG_HF_REPO"]
    assert pairs["LLAMA_ARG_HF_FILE"] == meta.runtime_env["LLAMA_ARG_HF_FILE"]
    assert pairs["LLM_DISABLE_THINKING"] == str(meta.settings.disable_thinking).lower()


@pytest.mark.parametrize("alias", sorted(RESULTS))
def test_blocks_leave_memory_to_the_server_table(alias: str) -> None:
    block = _block(alias)
    assert "LLM_MEM_LIMIT" not in _pairs(alias)
    assert "LLM_MEM_LIMIT" not in block
    assert (NOT_MEASURED in block) == (alias not in BENCHES)


def test_env_example_memory_limit_fits_the_default_model() -> None:
    env = env_example()
    need = llm_need_bytes(BENCHES[env["MODEL_ALIAS"]])
    assert need is not None
    assert _limit_bytes(env["LLM_MEM_LIMIT"]) >= MEMORY_MARGIN * need


@pytest.mark.parametrize("alias", sorted(BENCHES))
def test_long_input_timeout(alias: str) -> None:
    if _long_record_outcome(BENCHES[alias]) == "ok":
        return
    extra = json.loads((ROOT / "bench" / "extra" / f"{alias}-long-input.json").read_text("utf-8"))
    assert set(extra) == {"status", "time_total_s", "outcome"}
    timeout = _pairs(alias)["REQUEST_TIMEOUT_S"]
    assert float(timeout) >= LONG_INPUT_MARGIN * float(extra["time_total_s"])


def test_vps_table_matches_the_script() -> None:
    block = _between(README, "<!-- vps-table:start -->", "<!-- vps-table:end -->")
    assert block.strip() == recommend_text(ROOT).strip()


def test_recommended_model_is_the_env_example_model() -> None:
    matches = RECOMMENDED.findall(README)
    assert len(matches) == 1
    alias = matches[0]
    env = env_example()
    assert alias == env["MODEL_ALIAS"]
    for key, value in _pairs(alias).items():
        assert env.get(key) == value, key
