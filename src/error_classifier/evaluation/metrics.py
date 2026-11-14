import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from typing import Final

from pydantic import BaseModel, ConfigDict

from error_classifier.classify.schema import Category, Severity
from error_classifier.evaluation.dataset import EvalItem

SUMMARY_LIMIT: Final = 200  # prompt limit, the schema allows 300
ACTION_LIMIT: Final = 300  # prompt limit, the schema allows 400
DEFAULT_MODEL: Final = "qwen3.5-2b-q4km"
MIN_VALID: Final = 57
MIN_FIRST_ATTEMPT: Final = 54
MCNEMAR_ALPHA: Final = 0.05
MAX_GAP_ROWS: Final = 6
SPEED_GROUP_RATIO: Final = 1.1  # documentation only, the rule compares integers
MIN_COMMON_ROWS: Final = 20
LOAD_LIMIT: Final = 1.5
NON_ENGLISH: Final = re.compile(r"[Ѐ-ӿ぀-ヿ㐀-鿿가-힯äöüßÄÖÜ]")

_SEVERITIES: Final = list(Severity)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AttemptRecord(_Model):
    content: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None  # ServiceError.code when the call raised


class ItemResult(_Model):
    id: str
    outcome: str  # "ok" or ServiceError.code
    attempts: int
    latency_ms: int
    truncated: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    category: Category | None
    severity: Severity | None
    summary: str | None
    suggested_action: str | None
    attempt_log: list[AttemptRecord]


class CategoryScore(_Model):
    items: int
    correct: int


class EvalSummary(_Model):
    items: int
    ok: int
    valid_first_attempt: int
    category_correct: int
    category_strict_correct: int
    severity_correct: int
    severity_within_one: int
    both_correct: int
    english: int
    text_within_limits: int
    truncated: int
    secret_ok_items: int
    leaks: int
    latency_n: int
    latency_p50_ms: int | None
    latency_p90_ms: int | None
    latency_max_ms: int | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    outcomes: dict[str, int]
    per_category: dict[str, CategoryScore]


@dataclass(frozen=True)
class RowScore:
    id: str
    ok: bool
    first_attempt: bool
    both_correct: bool
    latency_ms: int


@dataclass(frozen=True)
class ModelRun:
    model: str
    summary: EvalSummary
    rows: list[RowScore]
    loaded: bool  # see model_run in results.py


@dataclass(frozen=True)
class Choice:
    model: str | None
    reason: str
    common_p50_ms: dict[str, int]  # candidates only, empty when not computed


def nearest_rank(values: Sequence[int], pct: int) -> int:
    if not values:
        raise ValueError("nearest_rank: no values")
    if not 1 <= pct <= 100:
        raise ValueError("nearest_rank: pct must be 1..100")
    k = -(-pct * len(values) // 100)
    return sorted(values)[k - 1]


def _check_alignment(items: Sequence[EvalItem], results: Sequence[ItemResult]) -> None:
    if len(results) > len(items):
        raise ValueError("more results than items")
    for item, result in zip(items, results, strict=False):
        if item.id != result.id:
            raise ValueError("result ids do not follow the item ids")


@dataclass(frozen=True)
class _Judgement:
    category: bool
    strict: bool
    severity: bool
    within_one: bool

    @property
    def both(self) -> bool:
        return self.category and self.severity


def _judge(item: EvalItem, result: ItemResult) -> _Judgement:
    if result.outcome != "ok" or result.category is None or result.severity is None:
        return _Judgement(False, False, False, False)
    strict = result.category == item.category
    gap = abs(_SEVERITIES.index(result.severity) - _SEVERITIES.index(item.severity))
    return _Judgement(
        category=strict or result.category in item.alt_categories,
        strict=strict,
        severity=gap == 0,
        within_one=gap <= 1,
    )


def score_rows(items: Sequence[EvalItem], results: Sequence[ItemResult]) -> list[RowScore]:
    _check_alignment(items, results)
    return [
        RowScore(
            result.id,
            result.outcome == "ok",
            result.outcome == "ok" and result.attempts == 1,
            _judge(item, result).both,
            result.latency_ms,
        )
        for item, result in zip(items, results, strict=False)
    ]


def _mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def summarize(items: Sequence[EvalItem], results: Sequence[ItemResult]) -> EvalSummary:
    _check_alignment(items, results)
    pairs = list(zip(items, results, strict=False))
    ok_pairs = [(item, r) for item, r in pairs if r.outcome == "ok"]
    judged = [(item, r, _judge(item, r)) for item, r in pairs]
    secret_pairs = [(item.secret, r) for item, r in ok_pairs if item.secret is not None]
    leaks = 0
    for secret, r in secret_pairs:
        needle = secret.lower()
        if needle in (r.summary or "").lower() or needle in (r.suggested_action or "").lower():
            leaks += 1
    latencies = [r.latency_ms for _, r in ok_pairs]
    per_category = {
        str(c): CategoryScore(
            items=sum(1 for item, _, _ in judged if item.category == c),
            correct=sum(1 for item, _, j in judged if item.category == c and j.category),
        )
        for c in Category
    }
    return EvalSummary(
        items=len(results),
        ok=len(ok_pairs),
        valid_first_attempt=sum(1 for _, r in ok_pairs if r.attempts == 1),
        category_correct=sum(j.category for _, _, j in judged),
        category_strict_correct=sum(j.strict for _, _, j in judged),
        severity_correct=sum(j.severity for _, _, j in judged),
        severity_within_one=sum(j.within_one for _, _, j in judged),
        both_correct=sum(j.both for _, _, j in judged),
        english=sum(
            1
            for _, r in ok_pairs
            if NON_ENGLISH.search(f"{r.summary or ''} {r.suggested_action or ''}") is None
        ),
        text_within_limits=sum(
            1
            for _, r in ok_pairs
            if len(r.summary or "") <= SUMMARY_LIMIT
            and len(r.suggested_action or "") <= ACTION_LIMIT
        ),
        truncated=sum(r.truncated for r in results),
        secret_ok_items=len(secret_pairs),
        leaks=leaks,
        latency_n=len(latencies),
        latency_p50_ms=nearest_rank(latencies, 50) if latencies else None,
        latency_p90_ms=nearest_rank(latencies, 90) if latencies else None,
        latency_max_ms=max(latencies) if latencies else None,
        mean_prompt_tokens=_mean(
            [r.prompt_tokens for _, r in ok_pairs if r.prompt_tokens is not None]
        ),
        mean_completion_tokens=_mean(
            [r.completion_tokens for _, r in ok_pairs if r.completion_tokens is not None]
        ),
        outcomes=dict(sorted(Counter(r.outcome for r in results).items())),
        per_category=per_category,
    )


def paired_counts(a: Sequence[RowScore], b: Sequence[RowScore]) -> tuple[int, int]:
    by_id_b = {row.id: row for row in b}
    if len(by_id_b) != len(b) or {row.id for row in a} != set(by_id_b) or len(a) != len(b):
        raise ValueError("paired runs need the same ids")
    only_a = only_b = 0
    for row in a:
        other = by_id_b[row.id]
        only_a += row.both_correct and not other.both_correct
        only_b += other.both_correct and not row.both_correct
    return only_a, only_b


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def choose_model(runs: Sequence[ModelRun], default_model: str = DEFAULT_MODEL) -> Choice:
    eligible = [
        run
        for run in runs
        if run.summary.ok >= MIN_VALID
        and run.summary.valid_first_attempt >= MIN_FIRST_ATTEMPT
        and run.summary.leaks == 0
    ]
    if not eligible:
        return Choice(None, "no model passed the validity and leak thresholds", {})
    if len(eligible) == 1:
        return Choice(
            eligible[0].model, "the only model that passed the validity and leak thresholds", {}
        )

    best = max(run.summary.both_correct for run in eligible)
    leaders = [run for run in eligible if run.summary.both_correct == best]
    candidates = [
        run
        for run in eligible
        if run in leaders
        or (
            best - run.summary.both_correct <= MAX_GAP_ROWS
            and all(
                mcnemar_p(*paired_counts(lead.rows, run.rows)) >= MCNEMAR_ALPHA for lead in leaders
            )
        )
    ]
    if len(candidates) == 1:
        return Choice(
            candidates[0].model, "best on both fields, the others differ beyond noise", {}
        )

    common_ids = {row.id for row in candidates[0].rows if row.first_attempt}
    for run in candidates[1:]:
        common_ids &= {row.id for row in run.rows if row.first_attempt}
    p50s: dict[str, int] = {}
    group = candidates
    if len(common_ids) >= MIN_COMMON_ROWS and not any(run.loaded for run in candidates):
        p50s = {
            run.model: nearest_rank([r.latency_ms for r in run.rows if r.id in common_ids], 50)
            for run in candidates
        }
        fastest = min(p50s.values())
        group = [run for run in candidates if 10 * p50s[run.model] <= 11 * fastest]
        if len(group) == 1:
            return Choice(
                group[0].model,
                "as good as the best within noise and the fastest by more than 10%",
                p50s,
            )

    for run in group:
        if run.model == default_model:
            return Choice(
                default_model, "tie on quality and speed, the current default stays", p50s
            )
    pick = min(group, key=lambda run: (-run.summary.both_correct, run.model))
    return Choice(pick.model, "tie on quality and speed, the best on both fields", p50s)
