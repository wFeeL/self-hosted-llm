from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from error_classifier.benchmark.report import (
    BenchFile,
    llm_need_bytes,
    read_bench,
    render_bench_report,
)
from error_classifier.evaluation.dataset import DEFAULT_DATASET, load_dataset
from error_classifier.evaluation.metrics import ModelRun, choose_model
from error_classifier.evaluation.results import model_run, read_result, render_report

__all__ = [
    "DEFAULT_TIER_GIB",
    "SYSTEM_RESERVE_GIB",
    "VPS_TIERS_GIB",
    "TierChoice",
    "choose_per_tier",
    "llm_need_bytes",
    "recommend_text",
    "render_tiers",
]

VPS_TIERS_GIB: Final = (2, 4, 8, 16)
SYSTEM_RESERVE_GIB: Final = 0.75
DEFAULT_TIER_GIB: Final = 8
GIB: Final = 2**30


@dataclass(frozen=True)
class TierChoice:
    tier_gib: int
    fits: tuple[str, ...]
    model: str | None
    reason: str


def _memory_incomplete(bench: BenchFile) -> bool:
    return llm_need_bytes(bench) is None or bench.summary.memory.api_peak is None


def _fits(bench: BenchFile, tier_gib: int) -> bool:
    need = llm_need_bytes(bench)
    api_peak = bench.summary.memory.api_peak
    if need is None or api_peak is None:
        return False
    return need + api_peak + SYSTEM_RESERVE_GIB * GIB <= tier_gib * GIB


def choose_per_tier(runs: Sequence[ModelRun], benches: Sequence[BenchFile]) -> list[TierChoice]:
    by_model = {b.summary.config.model: b for b in benches}
    choices: list[TierChoice] = []
    for tier in VPS_TIERS_GIB:
        fitting = [
            run for run in runs if run.model in by_model and _fits(by_model[run.model], tier)
        ]
        if not fitting:
            choices.append(TierChoice(tier, (), None, "no measured model fits"))
            continue
        choice = choose_model(fitting)
        fits = tuple(sorted(run.model for run in fitting))
        choices.append(TierChoice(tier, fits, choice.model, choice.reason))
    return choices


def render_tiers(
    choices: Sequence[TierChoice], unmeasured: Sequence[str], incomplete: Sequence[str] = ()
) -> str:
    lines = [
        "| VPS RAM | Models that fit | Recommended | Reason |",
        "|---|---|---|---|",
        *(
            f"| {c.tier_gib} GiB | {', '.join(c.fits) or '-'} | {c.model or 'none'} | {c.reason} |"
            for c in choices
        ),
    ]
    text = "\n".join(lines) + "\n"
    if unmeasured:
        text += f"\nNot measured: {', '.join(unmeasured)}.\n"
    if incomplete:
        text += f"\nMemory incomplete: {', '.join(incomplete)}.\n"
    return text


def recommend_text(root: Path) -> str:
    items = load_dataset(root / DEFAULT_DATASET)
    docs = [read_result(p) for p in sorted((root / "eval" / "results").glob("*.json"))]
    benches = [read_bench(p) for p in sorted((root / "bench" / "results").glob("*.jsonl"))]
    # Rendering runs the consistency checks of both reports, the text is not used.
    render_report(docs, items)
    render_bench_report(benches)
    runs = [model_run(doc, items) for doc in docs]
    measured = {b.summary.config.model for b in benches}
    unmeasured = sorted(run.model for run in runs if run.model not in measured)
    incomplete = sorted(b.summary.config.model for b in benches if _memory_incomplete(b))
    return render_tiers(choose_per_tier(runs, benches), unmeasured, incomplete)
