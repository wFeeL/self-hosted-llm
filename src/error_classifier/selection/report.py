import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final

from pydantic import BaseModel

from error_classifier.benchmark.report import GIB
from error_classifier.selection.check import CandidateResult
from error_classifier.selection.config import SelectSettings
from error_classifier.selection.decide import Decision, Reference
from error_classifier.selection.presets import Candidate, parse_size
from error_classifier.selection.system import HostInfo

HEADROOM: Final = 3 * GIB // 4
MAX_QUEUE_TIMEOUT_S: Final = 600
COLUMNS: Final = (
    "Model",
    "Status",
    "Load, s",
    "Smoke ok",
    "Leaks",
    "p50, s",
    "p95, s",
    "Long input, s",
    "Need here, GiB",
    "VmHWM, GiB",
    "Reference both",
    "Agrees with reference",
    "Notes",
)
LIMITS: Final = (
    "The smoke catches only gross failures: a handful of rows says little about "
    "answer quality, so the quality ranking comes from the reference runs.",
    "The timeout hints are a judgement from one long input and a sequential smoke test, "
    "not a guarantee for parallel load.",
)


@dataclass(frozen=True)
class RunContext:
    host: HostInfo
    settings: SelectSettings
    started_at: str  # UTC "%Y-%m-%dT%H:%M:%SZ"


def _round_up_to_ten(milliseconds: int) -> int:
    # 10 * ceil(2 * seconds / 10), in integers so 425000 ms gives exactly 850.
    return 10 * -(-2 * milliseconds // 10000)


def env_block(candidate: Candidate, result: CandidateResult, settings: SelectSettings) -> str:
    lines = [
        f"MODEL_ALIAS={candidate.alias}",
        f"LLAMA_ARG_HF_REPO={candidate.hf_repo}",
        f"LLAMA_ARG_HF_FILE={candidate.hf_file}",
        f"LLM_DISABLE_THINKING={'true' if candidate.disable_thinking else 'false'}",
    ]
    if result.long_outcome == "ok" and result.long_ms is not None:
        request = _round_up_to_ten(result.long_ms)
        if request > settings.request_timeout_s:
            lines.append(f"REQUEST_TIMEOUT_S={request}")
    if result.p95_ms is not None:
        queue = min(_round_up_to_ten(result.p95_ms), MAX_QUEUE_TIMEOUT_S)
        if queue > settings.queue_timeout_s:
            lines.append(f"QUEUE_TIMEOUT_S={queue}")
    return "\n".join(lines)


def _gib(size: int | None) -> str:
    return "-" if size is None else f"{size / GIB:.2f}"


def _seconds(ms: int | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}"


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _agrees(result: CandidateResult, ref: Reference) -> str:
    if not ref.categories or result.smoke_n == 0:
        return "-"
    same = sum(
        1
        for row in result.results
        if row.category is not None and ref.categories.get(row.id) == str(row.category)
    )
    return f"{same}/{result.smoke_n}"


def _long_input(result: CandidateResult) -> str:
    if result.long_outcome == "ok":
        return _seconds(result.long_ms)
    return result.long_outcome or "-"


def _row(result: CandidateResult, ref: Reference, status: str) -> str:
    ran = result.smoke_n > 0
    notes = [*result.failures, *result.warnings]
    if result.skipped is not None:
        notes.insert(0, result.skipped)
    if ref.note is not None:
        notes.append(ref.note)
    both = "-"
    if ref.run is not None:
        both = f"{ref.run.summary.both_correct}/{ref.run.summary.items}"
    cells = [
        result.alias,
        status,
        "-" if result.load is None else f"{result.load.seconds:.0f}",
        f"{result.ok}/{result.smoke_n}" if ran else "-",
        str(result.leaks) if ran else "-",
        _seconds(result.p50_ms),
        _seconds(result.p95_ms),
        _long_input(result),
        _gib(result.need_bytes),
        _gib(result.vm_hwm),
        both,
        _agrees(result, ref),
        "; ".join(notes) or "-",
    ]
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _conditions(ctx: RunContext) -> str:
    host, s = ctx.host, ctx.settings
    total = "unknown" if host.mem_total is None else f"{host.mem_total / GIB:.2f} GiB"
    disk = "unknown" if host.disk_free is None else f"{host.disk_free / GIB:.1f} GiB"
    parts = [
        f"router {s.select_base_url}",
        f"CPUs {host.cpus}",
        f"CPU model {host.cpu_model or 'unknown'}",
        f"flags {', '.join(host.cpu_flags) or 'none of avx2, avx512f, asimd'}",
        f"MemTotal {total}",
        f"disk free {disk}",
        f"LLM_CPUS {s.llm_cpus}",
        f"LLM_MEM_LIMIT {s.llm_mem_limit}",
        f"threads {s.llama_arg_threads}",
        f"context {s.llama_arg_ctx_size}",
        f"slots {s.llama_arg_n_parallel}",
        "warm start",
        "one request at a time",
        "prompt cache off",
        "p95 is nearest rank over the smoke rows and excludes queueing",
    ]
    return "Conditions: " + "; ".join(parts) + "."


def _warnings(ctx: RunContext, results: Sequence[CandidateResult]) -> list[str]:
    total = ctx.host.mem_total
    if total is None:
        return []
    found: list[str] = []
    api = parse_size(ctx.settings.api_mem_limit)
    for result in results:
        need = result.need_bytes
        if need is not None and result.passed and need + api + HEADROOM > total:
            found.append(
                f"{result.alias}: needs {need / GIB:.2f} GiB, with API_MEM_LIMIT "
                f"{ctx.settings.api_mem_limit} and 0.75 GiB for the system that is above "
                f"MemTotal {total / GIB:.2f} GiB"
            )
    if parse_size(ctx.settings.llm_mem_limit) > total - HEADROOM:
        found.append(
            f"LLM_MEM_LIMIT {ctx.settings.llm_mem_limit} is above MemTotal "
            f"{total / GIB:.2f} GiB minus 0.75 GiB"
        )
    return found


def render_report(
    ctx: RunContext,
    candidates: Sequence[Candidate],
    results: Sequence[CandidateResult],
    refs: Mapping[str, Reference],
    decision: Decision,
) -> str:
    by_alias = {r.alias: r for r in results}
    lines = ["# Model selection", "", _conditions(ctx), ""]
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("|" + "---|" * len(COLUMNS))
    for candidate in candidates:
        result = by_alias.get(candidate.alias)
        if result is not None:
            status = decision.status.get(candidate.alias, "-")
            lines.append(_row(result, refs[candidate.alias], status))
    lines.append("")
    if decision.model is None:
        lines.append(f"No model chosen: {decision.reason}")
    else:
        lines.append(f"Chosen: {decision.model} ({decision.reason})")
        chosen = next(c for c in candidates if c.alias == decision.model)
        block = env_block(chosen, by_alias[decision.model], ctx.settings)
        lines += ["", "Put this into `.env`:", "", "```", block, "```"]
    warnings = _warnings(ctx, results)
    if warnings:
        lines += ["", "Warnings:", ""] + [f"- {w}" for w in warnings]
    lines += ["", *LIMITS, ""]
    return "\n".join(lines)


def _default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


def report_json(
    ctx: RunContext,
    results: Sequence[CandidateResult],
    refs: Mapping[str, Reference],
    decision: Decision,
) -> str:
    data = {
        "host": asdict(ctx.host),
        "settings": ctx.settings.model_dump(),
        "started_at": ctx.started_at,
        "results": [asdict(r) for r in results],
        "references": {
            alias: {
                "note": ref.note,
                "need_bytes": ref.need_bytes,
                "both_correct": None if ref.run is None else ref.run.summary.both_correct,
            }
            for alias, ref in refs.items()
        },
        "decision": asdict(decision),
    }
    return json.dumps(data, sort_keys=True, indent=2, default=_default)
