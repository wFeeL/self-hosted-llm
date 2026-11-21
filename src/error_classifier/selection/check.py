from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from error_classifier.classify.schema import Category
from error_classifier.classify.service import (
    MIN_RETRY_S,
    ChatClient,
    Classifier,
    ClassifierOptions,
    ClassifyStats,
)
from error_classifier.errors import LlmUnavailableError, ServiceError
from error_classifier.evaluation.dataset import EvalItem
from error_classifier.evaluation.metrics import ItemResult, nearest_rank, summarize
from error_classifier.evaluation.runner import run_eval
from error_classifier.llm.client import ChatCompletion, ChatMessage
from error_classifier.selection.presets import Candidate
from error_classifier.selection.router import LoadOutcome, ModelStatus
from error_classifier.selection.system import (
    ChildProcess,
    ProcessMemory,
    child_arg,
    find_child,
    model_file_size,
    read_memory,
)


def smoke_items(items: Sequence[EvalItem]) -> list[EvalItem]:
    """All rows with a secret, then the first row of every category they miss."""
    chosen = {i for i, item in enumerate(items) if item.secret is not None}
    seen = {items[i].category for i in chosen}
    for category in Category:
        if category in seen:
            continue
        for i, item in enumerate(items):
            if item.category == category:
                chosen.add(i)
                break
    return [items[i] for i in sorted(chosen)]


class ModelCheckingChat:
    """ChatClient wrapper that counts answers reported under another model name."""

    def __init__(self, inner: ChatClient, alias: str) -> None:
        self._inner = inner
        self._alias = alias
        self.mismatches = 0

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, object],
        schema_name: str,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        disable_thinking: bool,
    ) -> ChatCompletion:
        completion = await self._inner.chat(
            messages,
            schema=schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            disable_thinking=disable_thinking,
        )
        if completion.model != self._alias:
            self.mismatches += 1
        return completion


class Router(Protocol):
    async def load(self, alias: str, timeout_s: float) -> LoadOutcome: ...
    async def statuses(self) -> dict[str, ModelStatus]: ...


@dataclass(frozen=True)
class CheckLimits:
    load_timeout_s: float
    model_max_s: float
    target_p95_s: float


@dataclass(frozen=True)
class CandidateResult:
    alias: str
    skipped: str | None  # prefilter reason, then nothing else is set
    load: LoadOutcome | None
    smoke_n: int
    results: list[ItemResult]  # smoke rows that ran, without the warm-up
    stop_reason: str | None
    ok: int
    leaks: int
    mismatches: int
    p50_ms: int | None  # nearest rank over ok rows
    p95_ms: int | None
    long_outcome: str | None  # "ok" or ServiceError.code, None if not run
    long_ms: int | None
    rss_anon_max: int | None
    vm_hwm: int | None
    file_size: int | None
    need_bytes: int | None  # rss_anon_max + file_size when both known
    restarted: bool
    hf_file_seen: str | None  # --hf-file of the child, None if not visible
    warnings: list[str]
    failures: list[str]  # empty means passed

    @property
    def passed(self) -> bool:
        return self.skipped is None and not self.failures


def _empty(alias: str, *, skipped: str | None, load: LoadOutcome | None) -> CandidateResult:
    return CandidateResult(
        alias=alias,
        skipped=skipped,
        load=load,
        smoke_n=0,
        results=[],
        stop_reason=None,
        ok=0,
        leaks=0,
        mismatches=0,
        p50_ms=None,
        p95_ms=None,
        long_outcome=None,
        long_ms=None,
        rss_anon_max=None,
        vm_hwm=None,
        file_size=None,
        need_bytes=None,
        restarted=False,
        hf_file_seen=None,
        warnings=[],
        failures=[],
    )


def skipped_result(alias: str, reason: str) -> CandidateResult:
    return _empty(alias, skipped=reason, load=None)


def load_failed_result(alias: str, load: LoadOutcome) -> CandidateResult:
    failure = f"load: {load.reason or 'failed'}"
    return replace(_empty(alias, skipped=None, load=load), failures=[failure])


class _Memory:
    """Peak anonymous memory and the last high-water mark over the readings."""

    def __init__(self, proc: Path) -> None:
        self._proc = proc
        self.rss_anon_max: int | None = None
        self.vm_hwm: int | None = None

    def sample(self, child: ChildProcess | None) -> None:
        if child is None:
            return
        memory: ProcessMemory | None = read_memory(child.pid, self._proc)
        if memory is None:
            return
        if memory.rss_anon is not None:
            self.rss_anon_max = max(self.rss_anon_max or 0, memory.rss_anon)
        if memory.vm_hwm is not None:
            self.vm_hwm = memory.vm_hwm


async def check_candidate(
    candidate: Candidate,
    *,
    router: Router,
    chat: ChatClient,  # LlmClient(model=candidate.alias) against the router
    options: ClassifierOptions,  # from settings, disable_thinking from the candidate
    smoke: Sequence[EvalItem],
    long_text: str,
    limits: CheckLimits,
    proc: Path,
    models_dir: Path,
    clock: Callable[[], float],
    log: Callable[[str], None],
) -> CandidateResult:
    alias = candidate.alias
    n = len(smoke)
    load = await router.load(alias, limits.load_timeout_s)
    if not load.ok:
        log(f"{alias}: load failed: {load.reason}")
        return load_failed_result(alias, load)
    log(f"{alias}: loaded in {load.seconds:.0f} s")

    t0 = clock()
    child = find_child(alias, proc)
    hf_file_seen = child_arg(child, "--hf-file") if child is not None else None
    memory = _Memory(proc)
    memory.sample(child)

    checking = ModelCheckingChat(chat, alias)
    rows_done = 0

    def on_item(result: ItemResult) -> None:
        nonlocal rows_done
        rows_done += 1
        log(f"{alias}: row {rows_done}/{n} {result.outcome} {result.latency_ms / 1000:.1f} s")

    try:
        outcome = await run_eval(
            smoke,
            checking,
            options,
            max_s=limits.model_max_s,
            probe=None,
            clock=clock,
            on_item=on_item,
        )
    except LlmUnavailableError:
        return _warmup_failed(alias, load, n, "runtime stopped during the warm-up", log)
    except ServiceError as exc:
        return _warmup_failed(alias, load, n, f"warm-up failed: {exc.code}", log)

    results = outcome.results
    stop_reason = outcome.stop_reason
    memory.sample(find_child(alias, proc) or child)

    long_outcome: str | None = None
    long_ms: int | None = None
    long_failure: str | None = None
    if stop_reason not in ("llm_unavailable", "llm_bad_response"):
        remaining = limits.model_max_s - (clock() - t0)
        if remaining < MIN_RETRY_S:
            long_failure = "no time left for the long input"
        else:
            classifier = Classifier(checking, replace(options, timeout_s=remaining), clock=clock)
            started = clock()
            try:
                await classifier.classify(long_text, ClassifyStats())
                long_outcome = "ok"
            except ServiceError as exc:
                long_outcome = exc.code
                long_failure = f"long input: {exc.code}"
            long_ms = round((clock() - started) * 1000)

    child2 = find_child(alias, proc)
    memory.sample(child2 or child)
    if child is not None:
        restarted = child2 is None or child2.pid != child.pid
    else:
        status = (await router.statuses()).get(alias)
        restarted = status is None or status.value != "loaded" or status.failed

    warnings: list[str] = []
    if child is None:
        warnings.append("runtime process not visible: the file and the memory were not verified")
    file_size = model_file_size(models_dir, candidate.hf_repo, candidate.hf_file)
    if file_size is None:
        warnings.append("model file not found in the models volume")
    need_bytes = (
        memory.rss_anon_max + file_size
        if memory.rss_anon_max is not None and file_size is not None
        else None
    )

    summary = summarize(smoke[: len(results)], results)
    ok_latencies = [r.latency_ms for r in results if r.outcome == "ok"]
    all_latencies = [r.latency_ms for r in results]
    p50_ms = nearest_rank(ok_latencies, 50) if ok_latencies else None
    # Errors stay in p95 so a request that ran into its timeout cannot hide.
    p95_ms = nearest_rank(all_latencies, 95) if all_latencies else None

    failures: list[str] = []
    if len(results) != n or stop_reason is not None:
        failures.append(
            f"smoke stopped after {len(results)} of {n} rows ({stop_reason or 'unknown'})"
        )
    if summary.ok < n - 1:
        failures.append(f"valid answers {summary.ok} of {n}")
    if summary.leaks > 0:
        failures.append(f"leaked {summary.leaks} secret(s)")
    if checking.mismatches > 0:
        failures.append("answers came from another model")
    if hf_file_seen is not None and hf_file_seen != candidate.hf_file:
        failures.append("the runtime loaded another file")
    if restarted:
        failures.append("the runtime restarted (likely out of memory)")
    if p95_ms is None:
        failures.append("p95 unknown")
    elif p95_ms > limits.target_p95_s * 1000:
        failures.append(f"p95 {p95_ms / 1000:.1f} s above {limits.target_p95_s:g} s")
    if long_failure is not None:
        failures.append(long_failure)

    log(f"{alias}: {failures[0] if failures else 'passed'}")
    return CandidateResult(
        alias=alias,
        skipped=None,
        load=load,
        smoke_n=n,
        results=results,
        stop_reason=stop_reason,
        ok=summary.ok,
        leaks=summary.leaks,
        mismatches=checking.mismatches,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        long_outcome=long_outcome,
        long_ms=long_ms,
        rss_anon_max=memory.rss_anon_max,
        vm_hwm=memory.vm_hwm,
        file_size=file_size,
        need_bytes=need_bytes,
        restarted=restarted,
        hf_file_seen=hf_file_seen,
        warnings=warnings,
        failures=failures,
    )


def _warmup_failed(
    alias: str, load: LoadOutcome, n: int, failure: str, log: Callable[[str], None]
) -> CandidateResult:
    log(f"{alias}: {failure}")
    return replace(_empty(alias, skipped=None, load=load), smoke_n=n, failures=[failure])
