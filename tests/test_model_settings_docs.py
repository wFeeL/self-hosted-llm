import re

from docs_helpers import ROOT, env_example

from error_classifier.benchmark.report import llm_need_bytes, read_bench
from error_classifier.selection.presets import load_presets

README = (ROOT / "README.md").read_text(encoding="utf-8")
BENCHES = {
    bench.summary.config.model: bench
    for bench in map(read_bench, sorted((ROOT / "bench/results").glob("*.jsonl")))
}
RECOMMENDED = re.compile(r"^Рекомендуется: \x60([^\x60]+)\x60$", re.MULTILINE)
LIMIT = re.compile(r"^(\d+)([gm])$")
LONG_INPUT_MARGIN = 1.3
# The peak was taken under the limit it is compared with, so it understates the need.
MEMORY_MARGIN = 1.1


def _limit_bytes(value: str) -> int:
    match = LIMIT.fullmatch(value)
    assert match is not None, value
    return int(match.group(1)) * (2**30 if match.group(2) == "g" else 2**20)


def test_env_example_memory_limit_fits_the_default_model() -> None:
    env = env_example()
    need = llm_need_bytes(BENCHES[env["MODEL_ALIAS"]])
    assert need is not None
    assert _limit_bytes(env["LLM_MEM_LIMIT"]) >= MEMORY_MARGIN * need


def test_env_example_timeout_covers_the_long_input() -> None:
    env = env_example()
    alias = env["MODEL_ALIAS"]
    paths = [
        *sorted((ROOT / "bench" / "results").glob("*.jsonl")),
        *sorted((ROOT / "bench" / "results-m2").glob("*.jsonl")),
    ]
    benches = [b for b in map(read_bench, paths) if b.summary.config.model == alias]
    assert benches
    times: list[float] = []
    for bench in benches:
        long_records = [r for r in bench.records if r.phase == "long"]
        assert long_records
        assert {r.outcome for r in long_records} == {"ok"}
        times += [r.wall_ms / 1000 for r in long_records]
    assert float(env["REQUEST_TIMEOUT_S"]) >= LONG_INPUT_MARGIN * max(times)


def test_recommended_model_is_the_env_example_model() -> None:
    matches = RECOMMENDED.findall(README)
    assert len(matches) == 1
    alias = matches[0]
    env = env_example()
    assert alias == env["MODEL_ALIAS"]
    preset = next(c for c in load_presets(ROOT / "select" / "presets.ini") if c.alias == alias)
    assert preset.hf_repo == env["LLAMA_ARG_HF_REPO"]
    assert preset.hf_file == env["LLAMA_ARG_HF_FILE"]
    assert str(preset.disable_thinking).lower() == env["LLM_DISABLE_THINKING"]
