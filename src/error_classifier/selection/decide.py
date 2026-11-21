import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from error_classifier.benchmark.report import GIB, llm_need_bytes, read_bench
from error_classifier.evaluation.dataset import DEFAULT_DATASET, EvalItem
from error_classifier.evaluation.metrics import ModelRun, choose_model, passes_thresholds
from error_classifier.evaluation.results import (
    dataset_sha256,
    model_run,
    prompt_sha256,
    read_result,
)
from error_classifier.selection.check import CandidateResult
from error_classifier.selection.presets import Candidate


@dataclass(frozen=True)
class Reference:
    run: ModelRun | None  # None when missing or stale
    note: str | None  # "no reference run", "reference is stale (prompt changed)", ...
    need_bytes: int | None  # llm_need_bytes of the reference benchmark
    categories: dict[str, str | None] = field(default_factory=dict[str, str | None])


@dataclass(frozen=True)
class Decision:
    model: str | None
    reason: str
    status: dict[str, str]


def _runs_file(env: Mapping[str, str], candidate: Candidate) -> bool:
    return (
        env.get("LLAMA_ARG_HF_REPO") == candidate.hf_repo
        and env.get("LLAMA_ARG_HF_FILE") == candidate.hf_file
    )


def _bench_needs(root: Path) -> dict[str, tuple[int | None, Mapping[str, str]]]:
    needs: dict[str, tuple[int | None, Mapping[str, str]]] = {}
    # File names start with the date, so the last one of a model is the newest.
    for path in sorted((root / "bench" / "results").glob("*.jsonl")):
        bench = read_bench(path)
        config = bench.summary.config
        needs[config.model] = (llm_need_bytes(bench), config.runtime_env)
    return needs


def _reference(
    candidate: Candidate,
    root: Path,
    items: Sequence[EvalItem],
    dataset_hash: str,
    need: int | None,
) -> Reference:
    path = root / "eval" / "results" / f"{candidate.alias}.json"
    if not path.exists():
        return Reference(None, "no reference run", need)
    doc = read_result(path)
    if not _runs_file(doc.meta.runtime_env, candidate):
        return Reference(None, "reference is for another file", need)
    if not doc.meta.complete:
        return Reference(None, "reference run is incomplete", need)
    if doc.meta.dataset_sha256 != dataset_hash:
        return Reference(None, "reference is stale (dataset changed)", need)
    if doc.meta.prompt_sha256 != prompt_sha256():
        return Reference(None, "reference is stale (prompt changed)", need)
    categories = {
        row.id: str(row.category) if row.category is not None else None for row in doc.items
    }
    return Reference(model_run(doc, items), None, need, categories)


def load_references(
    candidates: Sequence[Candidate], root: Path, items: Sequence[EvalItem]
) -> dict[str, Reference]:
    benches = _bench_needs(root)
    dataset_hash = dataset_sha256(root / DEFAULT_DATASET)
    refs: dict[str, Reference] = {}
    for candidate in candidates:
        need: int | None = None
        bench = benches.get(candidate.alias)
        if bench is not None and _runs_file(bench[1], candidate):
            need = bench[0]
        refs[candidate.alias] = _reference(candidate, root, items, dataset_hash, need)
    return refs


def prefilter(candidate: Candidate, ref: Reference, mem_limit: int, check_all: bool) -> str | None:
    if check_all or ref.need_bytes is None or ref.need_bytes <= mem_limit:
        return None
    return f"reference needs {ref.need_bytes / GIB:.2f} GiB, above LLM_MEM_LIMIT"


def _status(result: CandidateResult) -> str:
    if result.skipped is not None:
        return "skipped"
    return "passed" if result.passed else "failed"


def decide(results: Sequence[CandidateResult], refs: Mapping[str, Reference]) -> Decision:
    status = {r.alias: _status(r) for r in results}
    passed = [r for r in results if r.passed]
    chosen: str | None = None
    if not passed:
        reason = "no model passed the checks"
        if results and all(r.skipped is not None for r in results):
            reason += (
                ": all models were skipped by the reference memory; "
                "use --all or raise LLM_MEM_LIMIT"
            )
        return Decision(None, reason, status)
    if any(refs[r.alias].run is not None for r in results):
        pool: list[tuple[CandidateResult, ModelRun]] = []
        for r in passed:
            run = refs[r.alias].run
            if run is None:
                status[r.alias] = "passed, no reference"
            else:
                pool.append((r, run))
        eligible = [(r, run) for r, run in pool if passes_thresholds(run)]
        for r, run in pool:
            if not passes_thresholds(run):
                status[r.alias] = "reference failed thresholds"
        if not eligible:
            return Decision(
                None, "no passed model has a reference that meets the thresholds", status
            )
        choice = choose_model([run for _, run in eligible])
        chosen = next((r.alias for r, run in eligible if run.model == choice.model), None)
        reason = choice.reason
    else:
        chosen = min(
            passed, key=lambda r: (r.p50_ms if r.p50_ms is not None else sys.maxsize, r.alias)
        ).alias
        reason = "no reference quality; the fastest passed model by p50 on this server"
    if chosen is None:
        return Decision(None, reason, status)
    status[chosen] = "chosen"
    return Decision(chosen, reason, status)
