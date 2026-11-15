import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from error_classifier.classify.prompt import CUT_OFF_CORRECTION, build_messages, correction_for
from error_classifier.classify.schema import SCHEMA_NAME, Category, response_schema
from error_classifier.evaluation.dataset import EvalItem
from error_classifier.evaluation.metrics import (
    LOAD_LIMIT,
    EvalSummary,
    ItemResult,
    ModelRun,
    choose_model,
    mcnemar_p,
    paired_counts,
    score_rows,
)

RUNTIME_ENV: Final[tuple[str, ...]] = (
    "LLAMA_ARG_HF_REPO",
    "LLAMA_ARG_HF_FILE",
    "LLAMA_ARG_ALIAS",
    "LLAMA_ARG_CTX_SIZE",
    "LLAMA_ARG_N_PARALLEL",
    "LLAMA_ARG_THREADS",
    "LLAMA_ARG_BATCH",
    "LLAMA_ARG_REASONING",
    "LLAMA_ARG_CHAT_TEMPLATE_KWARGS",
    "LLAMA_ARG_MMPROJ_AUTO",
)

# Everything a fair comparison needs equal, except the model files and the alias.
SAME_RUNTIME: Final[tuple[str, ...]] = (
    "LLAMA_ARG_CTX_SIZE",
    "LLAMA_ARG_N_PARALLEL",
    "LLAMA_ARG_THREADS",
    "LLAMA_ARG_BATCH",
    "LLAMA_ARG_REASONING",
    "LLAMA_ARG_CHAT_TEMPLATE_KWARGS",
    "LLAMA_ARG_MMPROJ_AUTO",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSettings(_Model):
    temperature: float
    max_output_tokens: int
    max_input_chars: int
    request_timeout_s: float
    disable_thinking: bool


class RunMeta(_Model):
    model: str  # LLM_MODEL
    engine: Literal["podman", "docker"]
    container: str
    runtime_env: dict[str, str]  # RUNTIME_ENV found in the container env, D12
    runtime_build: str | None  # /props build_info
    model_path: str | None  # /props model_path
    total_slots: int | None  # /props total_slots
    slot_ctx: int | None  # /props default_generation_settings.n_ctx
    settings: RunSettings
    dataset_sha256: str
    prompt_sha256: str
    git_sha: str | None
    git_dirty: bool | None
    started_at: str  # UTC "%Y-%m-%dT%H:%M:%SZ"
    wall_s: float  # round(x, 1)
    loadavg_before: list[float] | None  # os.getloadavg(), rounded to 2
    loadavg_after: list[float] | None
    complete: bool
    stop_reason: str | None
    cpu_count: int | None
    machine: str


class ResultDoc(_Model):
    format: Literal[1]
    meta: RunMeta
    summary: EvalSummary
    items: list[ItemResult]


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_sha256() -> str:
    canonical = json.dumps(
        {
            "messages": build_messages(""),
            "schema": response_schema(),
            "schema_name": SCHEMA_NAME,
            "cut_off": CUT_OFF_CORRECTION,
            "correction": correction_for(["{reasons}"]),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def git_state(repo: Path) -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return None, None
    if head.returncode != 0 or status.returncode != 0:
        return None, None
    return head.stdout.strip(), bool(status.stdout.strip())


def container_env(engine: str, container: str) -> dict[str, str] | None:
    try:
        proc = subprocess.run(
            [engine, "inspect", "--format", "{{json .Config.Env}}", container],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        data: object = json.loads(proc.stdout)
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    env: dict[str, str] = {}
    for entry in data:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(entry, str):
            return None
        key, sep, value = entry.partition("=")
        if sep and key in RUNTIME_ENV:
            env[key] = value
    return env


def write_result(path: Path, doc: ResultDoc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_result(path: Path) -> ResultDoc:
    return ResultDoc.model_validate_json(path.read_text(encoding="utf-8"))


def model_run(doc: ResultDoc, items: Sequence[EvalItem]) -> ModelRun:
    meta = doc.meta
    # llama-server's own load after the run is about 10, so only the start counts.
    before = meta.loadavg_before
    loaded = before is not None and before[0] >= LOAD_LIMIT
    rows = score_rows(items[: len(doc.items)], doc.items)
    return ModelRun(meta.model, doc.summary, rows, loaded)


def _pct(k: int, n: int) -> str:
    return "-" if n == 0 else f"{k} ({round(100 * k / n)}%)"


def _seconds(ms: int | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}"


def _line(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    separator = "|" + "---|" * len(header)
    return "\n".join([_line(header), separator, *(_line(row) for row in rows)])


def render_report(docs: Sequence[ResultDoc], items: Sequence[EvalItem]) -> str:
    if not docs:
        raise ValueError("no results to report")
    ordered = sorted(docs, key=lambda d: d.meta.model)
    for doc in ordered:
        if not doc.meta.complete:
            raise ValueError(f"{doc.meta.model}: the run is incomplete")
    if len({d.meta.dataset_sha256 for d in ordered}) != 1:
        raise ValueError("the results come from different datasets")
    if len({d.meta.prompt_sha256 for d in ordered}) != 1:
        raise ValueError("the results come from different prompts")
    shared = [d.meta.settings.model_copy(update={"disable_thinking": False}) for d in ordered]
    if any(item != shared[0] for item in shared):
        raise ValueError("the results come from different settings")
    for key in SAME_RUNTIME:
        if len({d.meta.runtime_env.get(key) for d in ordered}) != 1:
            raise ValueError(f"the results come from different runtime settings: {key}")
    first_items = {c: s.items for c, s in ordered[0].summary.per_category.items()}
    for doc in ordered:
        if {c: s.items for c, s in doc.summary.per_category.items()} != first_items:
            raise ValueError("the results have different category counts")

    main_rows: list[list[str]] = []
    for doc in ordered:
        s = doc.summary
        main_rows.append(
            [
                doc.meta.model,
                str(s.items),
                _pct(s.valid_first_attempt, s.items),
                _pct(s.ok, s.items),
                _pct(s.category_correct, s.items),
                _pct(s.category_strict_correct, s.items),
                _pct(s.severity_correct, s.items),
                _pct(s.severity_within_one, s.items),
                _pct(s.both_correct, s.items),
                _pct(s.english, s.ok),
                _pct(s.text_within_limits, s.ok),
                f"{s.leaks}/{s.secret_ok_items}",
                f"{s.echoed}/{s.secret_ok_items}",
                _seconds(s.latency_p50_ms),
                _seconds(s.latency_p90_ms),
                _seconds(s.latency_max_ms),
                str(s.latency_n),
            ]
        )
    parts = [
        _table(
            [
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
                "Echoed by the model",
                "p50, s",
                "p90, s",
                "Max, s",
                "Latency n",
            ],
            main_rows,
        )
    ]

    codes = sorted({code for d in ordered for code in d.summary.outcomes} - {"ok"})
    codes = ["ok", *codes]
    parts.append(
        _table(
            ["Model", *codes],
            [[d.meta.model, *(str(d.summary.outcomes.get(c, 0)) for c in codes)] for d in ordered],
        )
    )

    models = [d.meta.model for d in ordered]
    parts.append(
        _table(
            ["Category", "Items", *models],
            [
                [
                    str(c),
                    str(first_items[str(c)]),
                    *(str(d.summary.per_category[str(c)].correct) for d in ordered),
                ]
                for c in Category
            ],
        )
    )

    runs = [model_run(d, items) for d in ordered]
    if len(runs) > 1:
        pair_rows: list[list[str]] = []
        for a, b in combinations(runs, 2):
            only_a, only_b = paired_counts(a.rows, b.rows)
            pair_rows.append(
                [
                    f"{a.model} vs {b.model}",
                    str(only_a),
                    str(only_b),
                    f"{mcnemar_p(only_a, only_b):.3f}",
                ]
            )
        parts.append(
            _table(["Pair", "Both only first", "Both only second", "p (McNemar)"], pair_rows)
        )
        k = len(pair_rows)
        noun = "pair" if k == 1 else "pairs"
        parts.append(
            f"p values are not corrected for multiple comparisons: {k} {noun}; "
            f"with Holm's correction the smallest p is significant only below {0.05 / k:.4f}."
        )

    choice = choose_model(runs)
    lines = [f"Choice by the rule: {choice.model or 'none'} ({choice.reason})."]
    if choice.common_p50_ms:
        lines.append(
            "p50 on common rows: "
            + ", ".join(f"{m} {ms / 1000:.1f} s" for m, ms in sorted(choice.common_p50_ms.items()))
            + "."
        )
    parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"
