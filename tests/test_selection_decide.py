from dataclasses import replace
from pathlib import Path

import pytest
from bench_files import GIB, bench_file

from error_classifier.benchmark.report import write_bench
from error_classifier.classify.schema import Category, Severity
from error_classifier.evaluation.dataset import EvalItem
from error_classifier.evaluation.metrics import (
    AttemptRecord,
    EvalSummary,
    ItemResult,
    ModelRun,
    RowScore,
    summarize,
)
from error_classifier.evaluation.results import (
    ResultDoc,
    RunMeta,
    RunSettings,
    dataset_sha256,
    prompt_sha256,
    write_result,
)
from error_classifier.selection.check import CandidateResult, skipped_result
from error_classifier.selection.decide import (
    Reference,
    decide,
    load_references,
    prefilter,
)
from error_classifier.selection.presets import Candidate

ITEMS = [
    EvalItem(f"t-{i}", "text", Category.RUNTIME_ERROR, Severity.MEDIUM, "go", (), None, "en", "")
    for i in range(1, 4)
]


def _summary(ok: int, first: int, both: int, leaks: int) -> EvalSummary:
    return summarize([], []).model_copy(
        update={
            "items": 60,
            "ok": ok,
            "valid_first_attempt": first,
            "both_correct": both,
            "leaks": leaks,
        }
    )


def _run(model: str, both: int, *, leaks: int = 0, ok: int = 60) -> ModelRun:
    rows = [RowScore(f"r{i:02d}", True, True, i < both, 1000) for i in range(60)]
    return ModelRun(model, _summary(ok, 60, both, leaks), rows, False)


def _ref(model: str, both: int, *, leaks: int = 0, need: int | None = None) -> Reference:
    return Reference(_run(model, both, leaks=leaks), None, need)


NO_REF = Reference(None, "no reference run", None)


def _result(alias: str, *, p50_ms: int = 1000, failures: tuple[str, ...] = ()) -> CandidateResult:
    base = replace(skipped_result(alias, ""), skipped=None)
    return replace(base, smoke_n=4, ok=4, p50_ms=p50_ms, failures=list(failures))


def test_decide_chooses_passed_with_good_reference() -> None:
    decision = decide([_result("a")], {"a": _ref("a", 50)})
    assert decision.model == "a"
    assert decision.status == {"a": "chosen"}
    assert decision.reason == "the only model that passed the validity and leak thresholds"


def test_decide_prefers_the_better_reference() -> None:
    results = [_result("a"), _result("b")]
    decision = decide(results, {"a": _ref("a", 30), "b": _ref("b", 50)})
    assert decision.model == "b"
    assert decision.status == {"a": "passed", "b": "chosen"}


def test_decide_reference_with_a_leak_is_not_chosen() -> None:
    decision = decide([_result("granite")], {"granite": _ref("granite", 50, leaks=1)})
    assert decision.model is None
    assert decision.reason == "no passed model has a reference that meets the thresholds"
    assert decision.status == {"granite": "reference failed thresholds"}


def test_decide_leaky_reference_loses_to_a_clean_one() -> None:
    results = [_result("granite"), _result("clean")]
    refs = {"granite": _ref("granite", 55, leaks=1), "clean": _ref("clean", 40)}
    decision = decide(results, refs)
    assert decision.model == "clean"
    assert decision.status == {"granite": "reference failed thresholds", "clean": "chosen"}


def test_decide_model_without_reference_is_not_chosen_when_references_exist() -> None:
    results = [_result("fast", p50_ms=100), _result("slow", p50_ms=9000)]
    decision = decide(results, {"fast": NO_REF, "slow": _ref("slow", 40)})
    assert decision.model == "slow"
    assert decision.status == {"fast": "passed, no reference", "slow": "chosen"}


def test_decide_only_unreferenced_model_passed_while_references_exist() -> None:
    results = [_result("fast"), _result("bad", failures=("leaked 1 secret(s)",))]
    decision = decide(results, {"fast": NO_REF, "bad": _ref("bad", 40)})
    assert decision.model is None
    assert decision.reason == "no passed model has a reference that meets the thresholds"
    assert decision.status == {"fast": "passed, no reference", "bad": "failed"}


def test_decide_without_any_reference_takes_the_smaller_p50() -> None:
    results = [_result("b", p50_ms=2000), _result("a", p50_ms=3000), _result("c", p50_ms=2000)]
    decision = decide(results, {"a": NO_REF, "b": NO_REF, "c": NO_REF})
    assert decision.model == "b"
    assert decision.reason == "no reference quality; the fastest passed model by p50 on this server"
    assert decision.status == {"a": "passed", "b": "chosen", "c": "passed"}


def test_decide_nobody_passed() -> None:
    results = [_result("a", failures=("p95 unknown",)), skipped_result("b", "too big")]
    decision = decide(results, {"a": NO_REF, "b": NO_REF})
    assert decision.model is None
    assert decision.reason == "no model passed the checks"
    assert decision.status == {"a": "failed", "b": "skipped"}


def test_decide_names_the_alias_when_the_reference_model_differs() -> None:
    decision = decide([_result("a")], {"a": _ref("run-name", 50)})
    assert decision.model == "a"


def test_prefilter() -> None:
    candidate = Candidate("a", "org/a", "a.gguf", False)
    big = Reference(None, None, 8 * GIB)
    assert prefilter(candidate, big, 6 * GIB, check_all=True) is None
    assert prefilter(candidate, Reference(None, None, None), 6 * GIB, check_all=False) is None
    assert prefilter(candidate, Reference(None, None, 5 * GIB), 6 * GIB, check_all=False) is None
    assert prefilter(candidate, big, 6 * GIB, check_all=False) == (
        "reference needs 8.00 GiB, above LLM_MEM_LIMIT"
    )


def _meta(**overrides: object) -> RunMeta:
    fields: dict[str, object] = {
        "model": "a",
        "engine": "podman",
        "container": "llm-1",
        "runtime_env": {},
        "runtime_build": None,
        "model_path": None,
        "total_slots": None,
        "slot_ctx": None,
        "settings": RunSettings(
            temperature=0.0,
            max_output_tokens=256,
            max_input_chars=3000,
            request_timeout_s=120.0,
            disable_thinking=False,
        ),
        "dataset_sha256": "",
        "prompt_sha256": prompt_sha256(),
        "git_sha": None,
        "git_dirty": None,
        "started_at": "2026-11-14T10:00:00Z",
        "wall_s": 1.0,
        "loadavg_before": None,
        "loadavg_after": None,
        "complete": True,
        "stop_reason": None,
        "cpu_count": 8,
        "machine": "x86_64",
    }
    fields.update(overrides)
    return RunMeta.model_validate(fields)


def _answer(id: str, category: Category | None) -> ItemResult:
    ok = category is not None
    return ItemResult(
        id=id,
        outcome="ok" if ok else "model_output_invalid",
        attempts=1,
        latency_ms=1000,
        truncated=False,
        prompt_tokens=None,
        completion_tokens=None,
        category=category,
        severity=Severity.MEDIUM if ok else None,
        summary="It failed." if ok else None,
        suggested_action="Check." if ok else None,
        attempt_log=[
            AttemptRecord(
                content="{}", finish_reason="stop", prompt_tokens=1, completion_tokens=1, error=None
            )
        ],
    )


class Root:
    def __init__(self, path: Path) -> None:
        self.path = path
        (path / "eval" / "results").mkdir(parents=True)
        (path / "eval" / "dataset.jsonl").write_text("{}\n", encoding="utf-8")

    def write_reference(self, alias: str, **meta: object) -> None:
        results = [
            _answer("t-1", Category.RUNTIME_ERROR),
            _answer("t-2", Category.NETWORK_ERROR),
            _answer("t-3", None),
        ]
        fields: dict[str, object] = {
            "dataset_sha256": dataset_sha256(self.path / "eval" / "dataset.jsonl")
        }
        fields.update(meta)
        doc = ResultDoc(
            format=1, meta=_meta(**fields), summary=summarize(ITEMS, results), items=results
        )
        write_result(self.path / "eval" / "results" / f"{alias}.json", doc)


def test_load_references_no_file(tmp_path: Path) -> None:
    Root(tmp_path)
    ref = load_references(["a"], tmp_path, ITEMS)["a"]
    assert ref == Reference(None, "no reference run", None, {})


def test_load_references_incomplete(tmp_path: Path) -> None:
    Root(tmp_path).write_reference("a", complete=False)
    ref = load_references(["a"], tmp_path, ITEMS)["a"]
    assert ref.run is None
    assert ref.note == "reference run is incomplete"


def test_load_references_stale_prompt(tmp_path: Path) -> None:
    Root(tmp_path).write_reference("a", prompt_sha256="0" * 64)
    ref = load_references(["a"], tmp_path, ITEMS)["a"]
    assert ref.run is None
    assert ref.note == "reference is stale (prompt changed)"


def test_load_references_stale_dataset(tmp_path: Path) -> None:
    Root(tmp_path).write_reference("a", dataset_sha256="0" * 64)
    ref = load_references(["a"], tmp_path, ITEMS)["a"]
    assert ref.run is None
    assert ref.note == "reference is stale (dataset changed)"


def test_load_references_good(tmp_path: Path) -> None:
    Root(tmp_path).write_reference("a")
    ref = load_references(["a"], tmp_path, ITEMS)["a"]
    assert ref.run is not None
    assert ref.run.model == "a"
    assert ref.note is None
    assert ref.categories == {"t-1": "runtime_error", "t-2": "network_error", "t-3": None}


def test_load_references_takes_the_newest_bench(tmp_path: Path) -> None:
    root = Root(tmp_path)
    root.write_reference("a")
    bench = tmp_path / "bench" / "results"
    for name, size in (
        ("2026-11-14-a.jsonl", GIB),
        ("2026-11-15-a.jsonl", 2 * GIB),
        ("2026-11-15-other.jsonl", GIB),
    ):
        model = "other" if "other" in name else "a"
        f = bench_file(model, conf={"model_file_size": size})
        write_bench(bench / name, f.records, f.readings, f.summary)
    refs = load_references(["a", "b"], tmp_path, ITEMS)
    assert refs["a"].need_bytes == 4 * GIB
    assert refs["b"].need_bytes is None
    assert refs["b"].note == "no reference run"


def test_load_references_broken_files_raise(tmp_path: Path) -> None:
    Root(tmp_path)
    (tmp_path / "eval" / "results" / "a.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_references(["a"], tmp_path, ITEMS)


def test_load_references_broken_bench_raises(tmp_path: Path) -> None:
    Root(tmp_path)
    bench = tmp_path / "bench" / "results"
    bench.mkdir(parents=True)
    (bench / "2026-11-15-a.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expecting value"):
        load_references(["a"], tmp_path, ITEMS)
