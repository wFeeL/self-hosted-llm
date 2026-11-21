import json
from dataclasses import replace

from bench_files import GIB

from error_classifier.classify.schema import Category, Severity
from error_classifier.evaluation.metrics import ItemResult
from error_classifier.selection.check import CandidateResult, skipped_result
from error_classifier.selection.config import SelectSettings
from error_classifier.selection.decide import Decision, Reference
from error_classifier.selection.presets import Candidate
from error_classifier.selection.report import RunContext, env_block, render_report, report_json
from error_classifier.selection.router import LoadOutcome
from error_classifier.selection.system import HostInfo

HEADER = (
    "| Model | Status | Load, s | Smoke ok | Leaks | p50, s | p95, s | Long input, s "
    "| Need here, GiB | VmHWM, GiB | Reference both | Agrees with reference | Notes |"
)
A = Candidate("alpha", "org/alpha", "alpha.gguf", True)
B = Candidate("beta", "org/beta", "beta.gguf", False)


def _settings(**changes: object) -> SelectSettings:
    return SelectSettings(**changes)  # pyright: ignore[reportArgumentType]


def _host(mem_total: int | None = 16 * GIB) -> HostInfo:
    return HostInfo(6, "AMD EPYC 7282", ("avx2",), mem_total, 8 * GIB, 100 * GIB)


def _ctx(settings: SelectSettings | None = None, mem_total: int | None = 16 * GIB) -> RunContext:
    return RunContext(_host(mem_total), settings or _settings(), "2026-11-21T10:00:00Z")


def _row(id: str, category: Category | None) -> ItemResult:
    ok = category is not None
    return ItemResult(
        id=id,
        outcome="ok" if ok else "llm_timeout",
        attempts=1,
        latency_ms=1000,
        truncated=False,
        prompt_tokens=None,
        completion_tokens=None,
        category=category,
        severity=Severity.LOW if ok else None,
        summary="s" if ok else None,
        suggested_action="a" if ok else None,
        attempt_log=[],
    )


def _passed(alias: str, **changes: object) -> CandidateResult:
    base = replace(
        skipped_result(alias, ""),
        skipped=None,
        load=LoadOutcome(True, 41.6, None, None),
        smoke_n=2,
        results=[_row("t-1", Category.RUNTIME_ERROR), _row("t-2", Category.NETWORK_ERROR)],
        ok=2,
        p50_ms=1500,
        p95_ms=2500,
        long_outcome="ok",
        long_ms=9000,
        rss_anon_max=2 * GIB,
        vm_hwm=3 * GIB,
        file_size=GIB,
        need_bytes=3 * GIB,
    )
    return replace(base, **changes)


REFS = {
    "alpha": Reference(None, "no reference run", None),
    "beta": Reference(None, "reference is stale (prompt changed)", None),
}


def test_env_block_without_hints() -> None:
    block = env_block(A, _passed("alpha", long_ms=1000, p95_ms=1000), _settings())
    assert block.splitlines() == [
        "MODEL_ALIAS=alpha",
        "LLAMA_ARG_HF_REPO=org/alpha",
        "LLAMA_ARG_HF_FILE=alpha.gguf",
        "LLM_DISABLE_THINKING=true",
    ]


def test_env_block_raises_the_request_timeout() -> None:
    block = env_block(B, _passed("beta", long_ms=425000, p95_ms=1000), _settings())
    assert "LLM_DISABLE_THINKING=false" in block
    assert block.splitlines()[-1] == "REQUEST_TIMEOUT_S=850"
    assert "QUEUE_TIMEOUT_S" not in block


def test_env_block_queue_timeout_follows_p95_and_is_capped() -> None:
    block = env_block(A, _passed("alpha", long_ms=1000, p95_ms=30000), _settings())
    assert block.splitlines()[-1] == "QUEUE_TIMEOUT_S=60"
    capped = env_block(A, _passed("alpha", long_ms=1000, p95_ms=400000), _settings())
    assert capped.splitlines()[-1] == "QUEUE_TIMEOUT_S=600"


def test_env_block_queue_timeout_not_printed_when_not_greater() -> None:
    block = env_block(A, _passed("alpha", long_ms=1000, p95_ms=10000), _settings())
    assert "QUEUE_TIMEOUT_S" not in block


def test_env_block_no_hint_without_a_long_answer() -> None:
    result = _passed("alpha", long_outcome="llm_timeout", long_ms=500000, p95_ms=None)
    assert env_block(A, result, _settings()).count("TIMEOUT_S") == 0


def test_render_report_has_columns_and_choice() -> None:
    results = [
        _passed("alpha"),
        skipped_result("beta", "reference needs 8.00 GiB, above LLM_MEM_LIMIT"),
    ]
    refs = {
        "alpha": Reference(None, None, None),
        "beta": REFS["beta"],
    }
    refs["alpha"] = Reference(None, None, None, {"t-1": "runtime_error", "t-2": "unknown"})
    decision = Decision(
        "alpha",
        "no reference quality; the fastest passed model by p50 on this server",
        {"alpha": "chosen", "beta": "skipped"},
    )
    text = render_report(_ctx(), [A, B], results, refs, decision)
    lines = text.splitlines()
    assert lines[0] == "# Model selection"
    assert HEADER in lines
    row = next(line for line in lines if line.startswith("| alpha "))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[:12] == [
        "alpha",
        "chosen",
        "42",
        "2/2",
        "0",
        "1.5",
        "2.5",
        "9.0",
        "3.00",
        "3.00",
        "-",
        "1/2",
    ]
    skipped = next(line for line in lines if line.startswith("| beta "))
    assert "skipped" in skipped
    assert "reference needs 8.00 GiB, above LLM_MEM_LIMIT" in skipped
    assert "reference is stale (prompt changed)" in skipped
    assert (
        "Chosen: alpha (no reference quality; the fastest passed model by p50 on this server)"
        in lines
    )
    assert "MODEL_ALIAS=alpha" in text
    assert "LLAMA_ARG_HF_FILE=alpha.gguf" in text
    assert "AMD EPYC 7282" in text
    assert "smoke catches only gross failures" in text
    assert "judgement" in text
    assert "Warnings" not in text


def test_render_report_reference_both_column() -> None:
    from error_classifier.evaluation.metrics import ModelRun, RowScore, summarize

    summary = summarize([], []).model_copy(update={"items": 60, "both_correct": 51})
    run = ModelRun("alpha", summary, [RowScore("t-1", True, True, True, 1)], False)
    refs = {"alpha": Reference(run, None, None, {}), "beta": REFS["beta"]}
    decision = Decision("alpha", "r", {"alpha": "chosen", "beta": "failed"})
    text = render_report(_ctx(), [A], [_passed("alpha")], refs, decision)
    row = next(line for line in text.splitlines() if line.startswith("| alpha "))
    assert "| 51/60 |" in row


def test_render_report_without_a_choice() -> None:
    results = [_passed("alpha", failures=["leaked 1 secret(s)"])]
    decision = Decision("alpha", "x", {"alpha": "failed"})
    decision = Decision(None, "no model passed the checks", decision.status)
    text = render_report(_ctx(), [A], results, REFS, decision)
    assert "No model chosen: no model passed the checks" in text.splitlines()
    assert "MODEL_ALIAS" not in text
    assert "leaked 1 secret(s)" in text


def test_render_report_warns_about_the_memory_limit() -> None:
    decision = Decision("alpha", "r", {"alpha": "chosen"})
    settings = _settings(llm_mem_limit="6656m")
    text = render_report(_ctx(settings, mem_total=6 * GIB), [A], [_passed("alpha")], REFS, decision)
    assert "Warnings" in text
    assert "LLM_MEM_LIMIT" in text.split("Warnings")[1]
    ok = render_report(
        _ctx(settings, mem_total=7 * GIB + GIB // 2),
        [A],
        [_passed("alpha", need_bytes=GIB)],
        REFS,
        decision,
    )
    assert "Warnings" not in ok


def test_render_report_warns_when_the_model_does_not_fit_with_the_api() -> None:
    decision = Decision("alpha", "r", {"alpha": "chosen"})
    settings = _settings(llm_mem_limit="1g", api_mem_limit="512m")
    # 6 GiB + 0.5 GiB + 0.75 GiB is above 7 GiB
    result = _passed("alpha", need_bytes=6 * GIB)
    text = render_report(_ctx(settings, mem_total=7 * GIB), [A], [result], REFS, decision)
    assert "alpha: needs 6.00 GiB" in text


def test_report_json_round_trips() -> None:
    results = [_passed("alpha"), skipped_result("beta", "too big")]
    decision = Decision("alpha", "r", {"alpha": "chosen", "beta": "skipped"})
    data = json.loads(report_json(_ctx(), results, REFS, decision))
    assert sorted(data) == ["decision", "host", "references", "results", "settings", "started_at"]
    assert data["decision"]["model"] == "alpha"
    assert data["host"]["cpu_model"] == "AMD EPYC 7282"
    assert data["results"][0]["results"][0]["id"] == "t-1"
    assert data["settings"]["presets_path"] == "/app/select/presets.ini"
    assert data["references"]["beta"] == {
        "note": "reference is stale (prompt changed)",
        "need_bytes": None,
        "both_correct": None,
    }
    assert data["started_at"] == "2026-11-21T10:00:00Z"
