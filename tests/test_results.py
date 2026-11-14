import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from error_classifier.classify import prompt
from error_classifier.classify.schema import Category, Severity
from error_classifier.evaluation.dataset import EvalItem
from error_classifier.evaluation.metrics import AttemptRecord, ItemResult, summarize
from error_classifier.evaluation.results import (
    ResultDoc,
    RunMeta,
    RunSettings,
    git_state,
    model_run,
    prompt_sha256,
    read_result,
    render_report,
    write_result,
)


def _item(id: str, category: Category, severity: Severity) -> EvalItem:
    return EvalItem(id, "text", category, severity, "python", (), None, "en", "note")


ITEMS = [
    _item("t-001", Category.RUNTIME_ERROR, Severity.MEDIUM),
    _item("t-002", Category.NETWORK_ERROR, Severity.HIGH),
    _item("t-003", Category.UNKNOWN, Severity.LOW),
]


def _ok(
    id: str, category: Category, severity: Severity, latency_ms: int, attempts: int = 1
) -> ItemResult:
    return ItemResult(
        id=id,
        outcome="ok",
        attempts=attempts,
        latency_ms=latency_ms,
        truncated=False,
        prompt_tokens=100,
        completion_tokens=50,
        category=category,
        severity=severity,
        summary="The code failed.",
        suggested_action="Check the code.",
        attempt_log=[
            AttemptRecord(
                content="{}",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=50,
                error=None,
            )
        ],
    )


def _failed(id: str, latency_ms: int) -> ItemResult:
    return ItemResult(
        id=id,
        outcome="model_output_invalid",
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


A_RESULTS = [
    _ok("t-001", Category.RUNTIME_ERROR, Severity.MEDIUM, 1000),
    _ok("t-002", Category.RUNTIME_ERROR, Severity.HIGH, 3000, attempts=2),
    _ok("t-003", Category.UNKNOWN, Severity.LOW, 2000),
]
B_RESULTS = [
    _ok("t-001", Category.RUNTIME_ERROR, Severity.MEDIUM, 1500),
    _failed("t-002", 5000),
    _ok("t-003", Category.UNKNOWN, Severity.LOW, 2500),
]


def _meta(model: str, **overrides: object) -> RunMeta:
    fields: dict[str, object] = {
        "model": model,
        "engine": "podman",
        "container": "llm-1",
        "runtime_env": {"LLAMA_ARG_THREADS": "6"},
        "runtime_build": "b1",
        "model_path": "/models/m.gguf",
        "total_slots": 2,
        "slot_ctx": 4096,
        "settings": RunSettings(
            temperature=0.0,
            max_output_tokens=256,
            max_input_chars=3000,
            request_timeout_s=120.0,
            disable_thinking=False,
        ),
        "dataset_sha256": "d" * 64,
        "prompt_sha256": "p" * 64,
        "git_sha": None,
        "git_dirty": None,
        "started_at": "2026-11-14T10:00:00Z",
        "wall_s": 12.3,
        "loadavg_before": [0.5, 0.4, 0.3],
        "loadavg_after": [6.2, 5.0, 3.0],
        "complete": True,
        "stop_reason": None,
        "cpu_count": 8,
        "machine": "x86_64",
    }
    fields.update(overrides)
    return RunMeta.model_validate(fields)


def _doc(model: str, results: list[ItemResult], **meta: object) -> ResultDoc:
    return ResultDoc(
        format=1,
        meta=_meta(model, **meta),
        summary=summarize(ITEMS[: len(results)], results),
        items=results,
    )


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    doc = _doc("a-model", A_RESULTS)
    path = tmp_path / "out" / "a.json"
    write_result(path, doc)
    assert read_result(path) == doc
    assert sorted(p.name for p in path.parent.iterdir()) == ["a.json"]


def test_read_rejects_extra_field(tmp_path: Path) -> None:
    doc = _doc("a-model", A_RESULTS)
    path = tmp_path / "a.json"
    write_result(path, doc)
    text = path.read_text(encoding="utf-8").replace('"engine"', '"extra": 1, "engine"', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        read_result(path)


def test_prompt_hash_changes_with_few_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    before = prompt_sha256()
    assert before == prompt_sha256()
    monkeypatch.setattr(prompt, "FEW_SHOT", (("text", "answer"),))
    assert prompt_sha256() != before


def test_git_state_outside_a_repo(tmp_path: Path) -> None:
    assert git_state(tmp_path) == (None, None)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
    )


def test_git_state_in_a_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    sha, dirty = git_state(tmp_path)
    assert sha is not None
    assert len(sha) == 40
    assert set(sha) <= set("0123456789abcdef")
    assert dirty is False
    (tmp_path / "f.txt").write_text("two\n", encoding="utf-8")
    assert git_state(tmp_path) == (sha, True)


@pytest.mark.parametrize(
    ("before", "after", "env", "expected"),
    [
        ([1.5, 1.0, 1.0], [0.5, 0.5, 0.5], {"LLAMA_ARG_THREADS": "6"}, True),
        ([1.4, 1.0, 1.0], [7.4, 5.0, 3.0], {"LLAMA_ARG_THREADS": "6"}, False),
        ([1.4, 1.0, 1.0], [7.5, 5.0, 3.0], {"LLAMA_ARG_THREADS": "6"}, True),
        ([0.5, 0.5, 0.5], [5.5, 5.0, 3.0], {"LLAMA_ARG_THREADS": "4"}, True),
        ([0.5, 0.5, 0.5], [5.4, 5.0, 3.0], {"LLAMA_ARG_THREADS": "4"}, False),
        ([0.5, 0.5, 0.5], [7.5, 5.0, 3.0], {}, True),
        ([0.5, 0.5, 0.5], [7.4, 5.0, 3.0], {"LLAMA_ARG_THREADS": "many"}, False),
        (None, None, {}, False),
    ],
)
def test_model_run_loaded(
    before: list[float] | None,
    after: list[float] | None,
    env: dict[str, str],
    expected: bool,
) -> None:
    doc = _doc(
        "a-model",
        A_RESULTS,
        loadavg_before=before,
        loadavg_after=after,
        runtime_env=env,
    )
    run = model_run(doc, ITEMS)
    assert run.loaded is expected
    assert run.model == "a-model"
    assert run.summary == doc.summary
    assert [row.id for row in run.rows] == ["t-001", "t-002", "t-003"]


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _separator(columns: int) -> str:
    return "|" + "---|" * columns


EXPECTED_REPORT = "\n".join(
    [
        _row(
            "Model",
            "Items",
            "Valid, first try",
            "Valid",
            "Category",
            "Category, strict",
            "Severity",
            "Severity ±1",
            "Both",
            "English",
            "Text limits",
            "Leaks",
            "p50, s",
            "p90, s",
            "Max, s",
            "Latency n",
        ),
        _separator(16),
        _row(
            "a-model",
            "3",
            "2 (67%)",
            "3 (100%)",
            "2 (67%)",
            "2 (67%)",
            "3 (100%)",
            "3 (100%)",
            "2 (67%)",
            "3 (100%)",
            "3 (100%)",
            "0/0",
            "2.0",
            "3.0",
            "3.0",
            "3",
        ),
        _row(
            "b-model",
            "3",
            "2 (67%)",
            "2 (67%)",
            "2 (67%)",
            "2 (67%)",
            "2 (67%)",
            "2 (67%)",
            "2 (67%)",
            "2 (100%)",
            "2 (100%)",
            "0/0",
            "1.5",
            "2.5",
            "2.5",
            "2",
        ),
        "",
        _row("Model", "ok", "model_output_invalid"),
        _separator(3),
        _row("a-model", "3", "0"),
        _row("b-model", "2", "1"),
        "",
        _row("Category", "Items", "a-model", "b-model"),
        _separator(4),
        _row("syntax_error", "0", "0", "0"),
        _row("runtime_error", "1", "1", "1"),
        _row("validation_error", "0", "0", "0"),
        _row("network_error", "1", "0", "0"),
        _row("database_error", "0", "0", "0"),
        _row("dependency_error", "0", "0", "0"),
        _row("configuration_error", "0", "0", "0"),
        _row("resource_error", "0", "0", "0"),
        _row("permission_error", "0", "0", "0"),
        _row("unknown", "1", "1", "1"),
        "",
        _row("Pair", "Both only first", "Both only second", "p (McNemar)"),
        _separator(4),
        _row("a-model vs b-model", "0", "0", "1.000"),
        "",
        "Choice by rule D8: none (no model passed the validity and leak thresholds).",
        "",
    ]
)


def test_report_exact_text() -> None:
    docs = [_doc("b-model", B_RESULTS), _doc("a-model", A_RESULTS)]
    assert render_report(docs, ITEMS) == EXPECTED_REPORT


def test_report_single_document_has_no_pairs_block() -> None:
    text = render_report([_doc("a-model", A_RESULTS)], ITEMS)
    assert "Pair" not in text
    assert text.endswith("leak thresholds).\n")


def test_report_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no result"):
        render_report([], ITEMS)


def test_report_rejects_incomplete() -> None:
    docs = [_doc("a-model", A_RESULTS), _doc("b-model", B_RESULTS, complete=False)]
    with pytest.raises(ValueError, match="b-model: the run is incomplete"):
        render_report(docs, ITEMS)


def test_report_rejects_different_datasets() -> None:
    docs = [_doc("a-model", A_RESULTS), _doc("b-model", B_RESULTS, dataset_sha256="e" * 64)]
    with pytest.raises(ValueError, match="dataset"):
        render_report(docs, ITEMS)
