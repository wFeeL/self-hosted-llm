import json
import shutil
from pathlib import Path
from typing import cast

import pytest
from bench_files import GIB, bench_file

from error_classifier.benchmark.report import BenchFile, Memory, llm_need_bytes
from error_classifier.evaluation.metrics import EvalSummary, ModelRun, RowScore
from error_classifier.recommend import (
    DEFAULT_TIER_GIB,
    SYSTEM_RESERVE_GIB,
    VPS_TIERS_GIB,
    TierChoice,
    choose_per_tier,
    recommend_text,
    render_tiers,
)

ONLY = "the only model that passed the validity and leak thresholds"


def run(model: str, both: int = 40) -> ModelRun:
    summary = EvalSummary(
        items=60,
        ok=60,
        valid_first_attempt=60,
        category_correct=0,
        category_strict_correct=0,
        severity_correct=0,
        severity_within_one=0,
        both_correct=both,
        english=0,
        text_within_limits=0,
        truncated=0,
        secret_ok_items=0,
        leaks=0,
        echoed=0,
        latency_n=0,
        latency_p50_ms=None,
        latency_p90_ms=None,
        latency_max_ms=None,
        mean_prompt_tokens=None,
        mean_completion_tokens=None,
        outcomes={},
        per_category={},
    )
    rows = [RowScore(f"r{i:02d}", True, True, i < both, 1000) for i in range(60)]
    return ModelRun(model, summary, rows, False)


def bench(model: str, need: int, api_peak: int | None = 0) -> BenchFile:
    # anon plus GGUF is half of the need each, the container peak equals the need.
    return bench_file(
        model,
        conf={"model_file_size": need // 2},
        agg={"llm_anon_max": need - need // 2},
        memory=Memory(
            llm_peak=need, llm_anon=None, llm_file=None, api_peak=api_peak, api_anon=None
        ),
    )


def test_constants() -> None:
    assert VPS_TIERS_GIB == (2, 4, 8, 16)
    assert SYSTEM_RESERVE_GIB == 0.75
    assert DEFAULT_TIER_GIB == 8


def test_need_is_the_larger_of_conservative_and_peak() -> None:
    low_peak = bench_file(
        "a",
        conf={"model_file_size": GIB},
        agg={"llm_anon_max": 2 * GIB},
        memory=Memory(llm_peak=2 * GIB, llm_anon=None, llm_file=None, api_peak=None, api_anon=None),
    )
    high_peak = bench_file(
        "b",
        conf={"model_file_size": GIB},
        agg={"llm_anon_max": 2 * GIB},
        memory=Memory(llm_peak=5 * GIB, llm_anon=None, llm_file=None, api_peak=None, api_anon=None),
    )
    assert llm_need_bytes(low_peak) == 3 * GIB
    assert llm_need_bytes(high_peak) == 5 * GIB


def test_need_is_none_without_a_part() -> None:
    assert llm_need_bytes(bench_file("a", conf={"model_file_size": None})) is None
    assert llm_need_bytes(bench_file("a", agg={"llm_anon_max": None})) is None
    no_peak = Memory(llm_peak=None, llm_anon=None, llm_file=None, api_peak=None, api_anon=None)
    assert llm_need_bytes(bench_file("a", memory=no_peak)) is None


def test_need_exactly_on_the_border_fits() -> None:
    # 1 GiB + 0.25 GiB + 0.75 GiB reserve is exactly 2 GiB.
    choices = choose_per_tier([run("m")], [bench("m", GIB, api_peak=GIB // 4)])
    assert choices[0] == TierChoice(2, ("m",), "m", ONLY)


def test_one_byte_over_the_border_moves_to_the_next_tier() -> None:
    choices = choose_per_tier([run("m")], [bench("m", GIB, api_peak=GIB // 4 + 1)])
    assert choices[0] == TierChoice(2, (), None, "no measured model fits")
    assert choices[1].fits == ("m",)


def test_model_without_a_bench_file_never_fits() -> None:
    choices = choose_per_tier([run("m"), run("n")], [bench("m", GIB)])
    assert all(c.fits == ("m",) for c in choices)


def test_model_without_the_api_peak_never_fits() -> None:
    choices = choose_per_tier([run("m")], [bench("m", GIB, api_peak=None)])
    assert all(c.model is None for c in choices)


def test_empty_tier_has_no_recommendation() -> None:
    choices = choose_per_tier([run("m")], [bench("m", 3 * GIB)])
    assert [c.model for c in choices] == [None, "m", "m", "m"]
    assert choices[0].reason == "no measured model fits"


def test_fits_are_alphabetical_and_choice_comes_from_the_rule() -> None:
    runs = [run("b-big", 50), run("a-small", 30)]
    benches = [bench("b-big", 3 * GIB), bench("a-small", GIB // 2)]
    choices = choose_per_tier(runs, benches)
    assert choices[0].fits == ("a-small",)
    assert choices[1].fits == ("a-small", "b-big")
    assert choices[1].model == "b-big"
    assert choices[1].reason == (
        "best on both fields, the others are more than 6 rows behind or McNemar p < 0.05"
    )


def test_render_exact_text() -> None:
    choices = [
        TierChoice(2, (), None, "no measured model fits"),
        TierChoice(4, ("a", "b"), "b", "tie"),
    ]
    assert render_tiers(choices, ["c", "d"]) == (
        "| VPS RAM | Models that fit | Recommended | Reason |\n"
        "|---|---|---|---|\n"
        "| 2 GiB | - | none | no measured model fits |\n"
        "| 4 GiB | a, b | b | tie |\n"
        "\n"
        "Not measured: c, d.\n"
    )


def test_render_without_unmeasured_has_no_footer() -> None:
    text = render_tiers([TierChoice(2, ("a",), "a", "x")], [])
    assert text == (
        "| VPS RAM | Models that fit | Recommended | Reason |\n|---|---|---|---|\n"
        "| 2 GiB | a | a | x |\n"
    )


def test_render_lists_models_with_incomplete_memory_after_not_measured() -> None:
    text = render_tiers([TierChoice(2, ("a",), "a", "x")], ["c"], ["d", "e"])
    assert text.endswith("\nNot measured: c.\n\nMemory incomplete: d, e.\n")


def test_render_with_only_incomplete_memory() -> None:
    text = render_tiers([TierChoice(2, ("a",), "a", "x")], [], ["d"])
    assert text.endswith("| 2 GiB | a | a | x |\n\nMemory incomplete: d.\n")


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "eval").mkdir()
    shutil.copytree(ROOT / "eval" / "results", tmp_path / "eval" / "results")
    shutil.copy(ROOT / "eval" / "dataset.jsonl", tmp_path / "eval" / "dataset.jsonl")
    shutil.copytree(ROOT / "bench" / "results", tmp_path / "bench" / "results")
    return tmp_path


def _set_in_summary(path: Path, section: str, key: str, value: str | None) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    data = cast("dict[str, dict[str, object]]", json.loads(lines[-1]))
    data[section][key] = value
    lines[-1] = json.dumps(data)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_recommend_text_on_the_repository_files_works(tree: Path) -> None:
    assert recommend_text(tree) == recommend_text(ROOT)


def test_recommend_text_names_a_bench_file_with_incomplete_memory(tree: Path) -> None:
    target = next((tree / "bench" / "results").glob("*granite-4.1*"))
    _set_in_summary(target, "memory", "api_peak", None)

    text = recommend_text(tree)

    assert text.endswith("Memory incomplete: granite-4.1-3b-q4km.\n")
    assert "granite-4.1-3b-q4km," not in text


def test_recommend_text_refuses_evaluation_files_from_different_prompts(tree: Path) -> None:
    target = next((tree / "eval" / "results").glob("granite-4.1*.json"))
    data = json.loads(target.read_text(encoding="utf-8"))
    data["meta"]["prompt_sha256"] = "other"
    target.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="prompts"):
        recommend_text(tree)


def test_recommend_text_refuses_bench_files_from_different_inputs(tree: Path) -> None:
    target = next((tree / "bench" / "results").glob("*granite-4.1*"))
    _set_in_summary(target, "config", "inputs_sha256", "other")

    with pytest.raises(ValueError, match="inputs"):
        recommend_text(tree)
