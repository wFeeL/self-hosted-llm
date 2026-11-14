from collections.abc import Sequence

import pytest

from error_classifier.classify.schema import Category, Severity
from error_classifier.evaluation.dataset import EvalItem
from error_classifier.evaluation.metrics import (
    DEFAULT_MODEL,
    EvalSummary,
    ItemResult,
    ModelRun,
    RowScore,
    choose_model,
    mcnemar_p,
    nearest_rank,
    paired_counts,
    score_rows,
    summarize,
)


def _item(
    id: str,
    category: Category = Category.RUNTIME_ERROR,
    severity: Severity = Severity.MEDIUM,
    alt: tuple[Category, ...] = (),
    secret: str | None = None,
) -> EvalItem:
    return EvalItem(id, "text", category, severity, "python", alt, secret, "en", "note")


def _ok(
    id: str,
    category: Category = Category.RUNTIME_ERROR,
    severity: Severity = Severity.MEDIUM,
    *,
    attempts: int = 1,
    latency_ms: int = 1000,
    summary: str = "The code failed.",
    action: str = "Check the code.",
    truncated: bool = False,
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 50,
) -> ItemResult:
    return ItemResult(
        id=id,
        outcome="ok",
        attempts=attempts,
        latency_ms=latency_ms,
        truncated=truncated,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        category=category,
        severity=severity,
        summary=summary,
        suggested_action=action,
        attempt_log=[],
    )


def _err(id: str, outcome: str = "model_output_invalid", latency_ms: int = 2000) -> ItemResult:
    return ItemResult(
        id=id,
        outcome=outcome,
        attempts=2,
        latency_ms=latency_ms,
        truncated=False,
        prompt_tokens=None,
        completion_tokens=None,
        category=None,
        severity=None,
        summary=None,
        suggested_action=None,
        attempt_log=[],
    )


def _summary(ok: int, first: int, both: int, leaks: int = 0, items: int = 60) -> EvalSummary:
    return EvalSummary(
        items=items,
        ok=ok,
        valid_first_attempt=first,
        category_correct=0,
        category_strict_correct=0,
        severity_correct=0,
        severity_within_one=0,
        both_correct=both,
        english=0,
        text_within_limits=0,
        truncated=0,
        secret_ok_items=0,
        leaks=leaks,
        latency_n=0,
        latency_p50_ms=None,
        latency_p90_ms=None,
        latency_max_ms=None,
        mean_prompt_tokens=None,
        mean_completion_tokens=None,
        outcomes={},
        per_category={},
    )


def _run(
    model: str,
    both: set[int],
    ok: int,
    first: set[int],
    leaks: int,
    latencies: int | Sequence[int],
    *,
    loaded: bool = False,
    summary_first: int | None = None,
) -> ModelRun:
    rows = [
        RowScore(
            id=f"r{i:02d}",
            ok=i < ok,
            first_attempt=i in first,
            both_correct=i in both,
            latency_ms=latencies if isinstance(latencies, int) else latencies[i],
        )
        for i in range(60)
    ]
    first_count = len(first) if summary_first is None else summary_first
    return ModelRun(model, _summary(ok, first_count, len(both), leaks), rows, loaded)


def _good(model: str, both: int, latency: int, *, loaded: bool = False) -> ModelRun:
    return _run(model, set(range(both)), 60, set(range(60)), 0, latency, loaded=loaded)


ALL = set(range(60))


def test_nearest_rank_examples() -> None:
    values = [5, 1, 3, 2, 4]
    assert nearest_rank(values, 50) == 3
    assert nearest_rank(values, 90) == 5
    assert nearest_rank(values, 20) == 1
    sixty = list(range(1, 61))
    assert nearest_rank(sixty, 50) == 30
    assert nearest_rank(sixty, 90) == 54


@pytest.mark.parametrize(("values", "pct"), [([], 50), ([1], 0), ([1], 101)])
def test_nearest_rank_rejects_empty_and_bad_pct(values: list[int], pct: int) -> None:
    with pytest.raises(ValueError, match="nearest_rank"):
        nearest_rank(values, pct)


def test_error_row_counts_as_wrong() -> None:
    items = [_item("a-001"), _item("a-002")]
    s = summarize(items, [_ok("a-001"), _err("a-002")])
    assert s.items == 2
    assert s.ok == 1
    assert s.category_correct == 1
    assert s.both_correct == 1
    assert s.outcomes == {"model_output_invalid": 1, "ok": 1}


def test_alt_category_counts_only_lenient() -> None:
    items = [
        _item(
            "a-001",
            Category.CONFIGURATION_ERROR,
            Severity.HIGH,
            (Category.NETWORK_ERROR,),
        )
    ]
    s = summarize(items, [_ok("a-001", Category.NETWORK_ERROR, Severity.HIGH)])
    assert s.category_correct == 1
    assert s.category_strict_correct == 0
    assert s.both_correct == 1


def test_severity_within_one() -> None:
    items = [_item(f"a-00{i}", severity=Severity.HIGH) for i in (1, 2, 3)]
    results = [
        _ok("a-001", severity=Severity.MEDIUM),
        _ok("a-002", severity=Severity.LOW),
        _ok("a-003", severity=Severity.CRITICAL),
    ]
    s = summarize(items, results)
    assert s.severity_correct == 0
    assert s.severity_within_one == 2


def test_first_attempt_needs_one_attempt() -> None:
    items = [_item("a-001"), _item("a-002")]
    s = summarize(items, [_ok("a-001", attempts=1), _ok("a-002", attempts=2)])
    assert s.valid_first_attempt == 1
    assert s.ok == 2


def test_text_limits() -> None:
    items = [_item(f"a-00{i}") for i in (1, 2, 3)]
    results = [
        _ok("a-001", summary="x" * 201),
        _ok("a-002", action="x" * 301),
        _ok("a-003", summary="x" * 200, action="x" * 300),
    ]
    assert summarize(items, results).text_within_limits == 1


def test_english_check() -> None:
    items = [_item(f"a-00{i}") for i in (1, 2, 3)]
    results = [
        _ok("a-001", summary="Ошибка в коде."),
        _ok("a-002", action="Prüfen Sie die Datei."),
        _ok("a-003"),
    ]
    assert summarize(items, results).english == 1


def test_leak_is_case_insensitive_and_counts_only_ok() -> None:
    items = [_item("a-001", secret="Tr0ub4dor-77"), _item("a-002", secret="Tr0ub4dor-77")]
    results = [_ok("a-001", action="Rotate tr0ub4dor-77 now."), _err("a-002")]
    s = summarize(items, results)
    assert s.leaks == 1
    assert s.secret_ok_items == 1


def test_truncated_is_counted() -> None:
    items = [_item("a-001"), _item("a-002")]
    s = summarize(items, [_ok("a-001", truncated=True), _ok("a-002")])
    assert s.truncated == 1


def test_latency_only_from_ok_rows() -> None:
    items = [_item(f"a-00{i}") for i in (1, 2, 3)]
    results = [
        _ok("a-001", latency_ms=1000),
        _ok("a-002", latency_ms=3000),
        _err("a-003", latency_ms=90000),
    ]
    s = summarize(items, results)
    assert s.latency_n == 2
    assert s.latency_p50_ms == 1000
    assert s.latency_p90_ms == 3000
    assert s.latency_max_ms == 3000


def test_mean_tokens_skip_missing_values() -> None:
    items = [_item("a-001"), _item("a-002")]
    results = [
        _ok("a-001", prompt_tokens=100, completion_tokens=None),
        _ok("a-002", prompt_tokens=201, completion_tokens=40),
    ]
    s = summarize(items, results)
    assert s.mean_prompt_tokens == 150.5
    assert s.mean_completion_tokens == 40.0


def test_no_ok_rows_gives_none() -> None:
    s = summarize([_item("a-001")], [_err("a-001")])
    assert s.latency_n == 0
    assert s.latency_p50_ms is None
    assert s.latency_p90_ms is None
    assert s.latency_max_ms is None
    assert s.mean_prompt_tokens is None
    assert s.mean_completion_tokens is None


def test_per_category_lists_every_category() -> None:
    s = summarize([_item("a-001")], [_ok("a-001")])
    assert list(s.per_category) == [str(c) for c in Category]
    assert len(s.per_category) == 10
    assert (s.per_category["runtime_error"].items, s.per_category["runtime_error"].correct) == (
        1,
        1,
    )
    assert s.per_category["unknown"].items == 0
    assert s.per_category["unknown"].correct == 0


def test_mismatched_ids_raise() -> None:
    items = [_item("a-001")]
    with pytest.raises(ValueError, match="ids"):
        summarize(items, [_ok("a-002")])
    with pytest.raises(ValueError, match="ids"):
        score_rows(items, [_ok("a-002")])


def test_more_results_than_items_raise() -> None:
    items = [_item("a-001")]
    results = [_ok("a-001"), _ok("a-002")]
    with pytest.raises(ValueError, match="results"):
        summarize(items, results)
    with pytest.raises(ValueError, match="results"):
        score_rows(items, results)


def test_score_rows() -> None:
    items = [_item("a-001"), _item("a-002"), _item("a-003")]
    results = [_ok("a-001", latency_ms=500), _ok("a-002", attempts=2), _err("a-003")]
    rows = score_rows(items, results)
    assert rows == [
        RowScore("a-001", True, True, True, 500),
        RowScore("a-002", True, False, True, 1000),
        RowScore("a-003", False, False, False, 2000),
    ]


def _rows(both: set[int]) -> list[RowScore]:
    return [RowScore(f"r{i}", True, True, i in both, 1) for i in range(5)]


def test_paired_counts() -> None:
    assert paired_counts(_rows({1, 2}), _rows({2, 3})) == (1, 1)


def test_paired_counts_rejects_different_ids() -> None:
    other = [RowScore(f"x{i}", True, True, True, 1) for i in range(5)]
    with pytest.raises(ValueError, match="ids"):
        paired_counts(_rows({1}), other)


def test_mcnemar_p() -> None:
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(6, 0) == 0.03125
    assert mcnemar_p(5, 5) == 1.0
    assert mcnemar_p(9, 1) == pytest.approx(0.0215, rel=1e-3)


def test_choice_disqualifies() -> None:
    low_ok = _run("m-ok", set(range(40)), 56, ALL, 0, 1000)
    low_first = _run("m-first", set(range(40)), 60, set(range(53)), 0, 1000)
    leaky = _run("m-leak", set(range(40)), 60, ALL, 1, 1000)
    fine = _run("m-fine", set(range(10)), 60, ALL, 0, 1000)
    choice = choose_model([low_ok, low_first, leaky, fine])
    assert choice.model == "m-fine"


def test_choice_none_when_all_fail() -> None:
    runs = [_run(f"m-{i}", set(range(40)), 50, ALL, 0, 1000) for i in range(2)]
    choice = choose_model(runs)
    assert choice.model is None
    assert choice.reason == "no model passed the validity and leak thresholds"
    assert choice.common_p50_ms == {}


def test_choice_clear_quality_gap_wins() -> None:
    leader = _good("m-a", 30, 10000)
    other = _run("m-b", set(range(8, 30)), 60, ALL, 0, 7000)
    choice = choose_model([other, leader])
    assert choice.model == "m-a"
    assert choice.common_p50_ms == {}


def test_choice_six_rows_ahead_with_small_p_is_out() -> None:
    # 6:0 gives p 0.031, so the second model is out despite the gap of exactly six rows.
    leader = _good("m-a", 36, 10000)
    other = _run("m-b", set(range(6, 36)), 60, ALL, 0, 7000)
    choice = choose_model([leader, other])
    assert choice.model == "m-a"
    assert choice.common_p50_ms == {}


def test_choice_gap_above_six_rows_is_out() -> None:
    shared = set(range(29, 41))
    leader = _run("m-a", set(range(19)) | shared, 60, ALL, 0, 10000)
    other = _run("m-b", set(range(19, 29)) | shared, 60, ALL, 0, 7000)
    choice = choose_model([leader, other])
    assert choice.model == "m-a"
    assert choice.common_p50_ms == {}


def test_choice_six_rows_inside_noise_is_a_tie() -> None:
    shared = set(range(20, 40))
    leader = _run("m-a", set(range(13)) | shared, 60, ALL, 0, 10000)
    other = _run("m-b", set(range(13, 20)) | shared, 60, ALL, 0, 8000)
    choice = choose_model([leader, other])
    assert choice.model == "m-b"
    assert choice.common_p50_ms == {"m-a": 10000, "m-b": 8000}
    assert choice.reason == ("as good as the best within noise and the fastest by more than 10%")


def test_choice_speed_needs_more_than_10_percent() -> None:
    choice = choose_model([_good("m-a", 30, 10000), _good("m-b", 27, 9500)])
    assert choice.model == "m-a"
    assert choice.common_p50_ms == {"m-a": 10000, "m-b": 9500}
    assert choice.reason == "tie on quality and speed, the best on both fields"


def test_choice_tie_keeps_default() -> None:
    choice = choose_model([_good(DEFAULT_MODEL, 27, 10000), _good("m-x", 30, 9600)])
    assert choice.model == DEFAULT_MODEL
    assert choice.reason == "tie on quality and speed, the current default stays"


def test_choice_slow_default_loses_to_fast_pair() -> None:
    runs = [_good(DEFAULT_MODEL, 30, 13000), _good("m-a", 28, 10000), _good("m-b", 27, 10500)]
    choice = choose_model(runs)
    assert choice.model == "m-a"
    assert set(choice.common_p50_ms) == {DEFAULT_MODEL, "m-a", "m-b"}


def test_choice_name_breaks_the_last_tie() -> None:
    choice = choose_model([_good("m-b", 30, 10000), _good("m-a", 30, 10000)])
    assert choice.model == "m-a"


def test_choice_loaded_run_makes_speed_a_tie() -> None:
    choice = choose_model([_good("m-a", 30, 10000), _good("m-b", 28, 8000, loaded=True)])
    assert choice.model == "m-a"
    assert choice.common_p50_ms == {}


def test_choice_too_few_common_rows_is_a_tie() -> None:
    # Eligible models share at least 42 rows, so the summary and the rows are built
    # independently to reach the guard.
    narrow = _run("m-a", set(range(30)), 60, set(range(15)), 0, 10000, summary_first=60)
    fast = _run("m-b", set(range(28)), 60, ALL, 0, 7000)
    choice = choose_model([narrow, fast])
    assert choice.model == "m-a"
    assert choice.common_p50_ms == {}


def test_choice_common_rows_only() -> None:
    slow_latencies = [1000] * 6 + [5000] * 24 + [12000] * 30
    # On all rows m-a has p50 5000 and would win; on the 54 rows both answered first
    # time its p50 is 12000.
    slow = _run("m-a", set(range(30)), 60, ALL, 0, slow_latencies)
    other = _run("m-b", set(range(28)), 60, set(range(6, 60)), 0, 10000)
    choice = choose_model([slow, other])
    assert choice.common_p50_ms == {"m-a": 12000, "m-b": 10000}
    assert choice.model == "m-b"
