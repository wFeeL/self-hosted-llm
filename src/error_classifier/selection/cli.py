import argparse
import asyncio
import os
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TextIO

import httpx
from pydantic import ValidationError

from error_classifier.benchmark.runner import long_input
from error_classifier.classify.service import ClassifierOptions
from error_classifier.evaluation.dataset import DEFAULT_DATASET, EvalItem, load_dataset
from error_classifier.llm.client import LlmClient
from error_classifier.selection.check import (
    CandidateResult,
    CheckLimits,
    check_candidate,
    skipped_result,
    smoke_items,
)
from error_classifier.selection.config import SelectSettings
from error_classifier.selection.decide import (
    Decision,
    Reference,
    decide,
    load_references,
    prefilter,
)
from error_classifier.selection.presets import Candidate, load_presets, parse_size
from error_classifier.selection.report import RunContext, render_report, report_json
from error_classifier.selection.router import (
    RouterClient,
    RouterError,
    main_llm_running,
)
from error_classifier.selection.system import HostInfo, read_host

EXIT_CHOSEN, EXIT_CONFIG, EXIT_NONE, EXIT_ROUTER = 0, 2, 3, 4
CALL_TIMEOUT_S: Final = 10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="select",
        description="Check every model of the presets file on this server and choose one.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="check models even when their reference benchmark says they do not fit LLM_MEM_LIMIT",
    )
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="do not refuse to start while the main llm service answers",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        # The message of a validation error repeats the input values.
        return f"{exc.title} has {exc.error_count()} invalid field(s)"
    return str(exc)


def _emit(
    args: argparse.Namespace,
    ctx: RunContext,
    candidates: Sequence[Candidate],
    results: Sequence[CandidateResult],
    refs: dict[str, Reference],
    out: TextIO,
) -> Decision:
    decision = decide(results, refs)
    if args.json:
        text = report_json(ctx, results, refs, decision)
    else:
        text = render_report(ctx, candidates, results, refs, decision)
    out.write(text if text.endswith("\n") else text + "\n")
    return decision


async def _check_all(
    settings: SelectSettings,
    args: argparse.Namespace,
    router: RouterClient,
    candidates: Sequence[Candidate],
    refs: dict[str, Reference],
    items: Sequence[EvalItem],
    results: list[CandidateResult],
    *,
    mem_limit: int,
    proc: Path,
    models_dir: Path,
    clock: Callable[[], float],
    log: Callable[[str], None],
) -> bool:
    """Run every candidate; False means the router would not unload a model."""
    smoke = smoke_items(items)
    long_text = long_input()
    limits = CheckLimits(
        settings.select_load_timeout_s, settings.select_model_max_s, settings.select_target_p95_s
    )
    for candidate in candidates:
        alias = candidate.alias
        reason = prefilter(candidate, refs[alias], mem_limit, args.all)
        if reason is not None:
            log(f"{alias}: skipped ({reason})")
            results.append(skipped_result(alias, reason))
            continue
        options = ClassifierOptions(
            settings.max_input_chars,
            settings.max_output_tokens,
            settings.llm_temperature,
            settings.request_timeout_s,
            candidate.disable_thinking,
        )
        async with httpx.AsyncClient(
            base_url=settings.select_base_url + "/v1/", timeout=CALL_TIMEOUT_S
        ) as chat_http:
            result = await check_candidate(
                candidate,
                router=router,
                chat=LlmClient(chat_http, model=alias, probe_timeout_s=CALL_TIMEOUT_S),
                options=options,
                smoke=smoke,
                long_text=long_text,
                limits=limits,
                proc=proc,
                models_dir=models_dir,
                clock=clock,
                log=log,
            )
        results.append(result)
        if not await router.unload(alias):
            log(f"{alias}: the router did not unload the model in time")
            return False
    return True


async def run_selection(
    settings: SelectSettings,
    args: argparse.Namespace,
    *,
    proc: Path = Path("/proc"),
    models_dir: Path = Path("/models"),
    cpus: Callable[[], int] = lambda: len(os.sched_getaffinity(0)),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    def log(line: str) -> None:
        print(line, file=err)

    try:
        mem_limit = parse_size(settings.llm_mem_limit)
        parse_size(settings.api_mem_limit)
        candidates = load_presets(settings.presets_path)
        items = load_dataset(settings.data_root / DEFAULT_DATASET)
        refs = load_references([c.alias for c in candidates], settings.data_root, items)
    except (ValueError, OSError) as exc:
        log(f"configuration error: {_describe(exc)}")
        return EXIT_CONFIG

    async with httpx.AsyncClient() as probe:
        if not args.allow_running and await main_llm_running(probe, settings.main_llm_url):
            log("stop the main stack first: docker compose stop")
            return EXIT_CONFIG

    results: list[CandidateResult] = []
    host: HostInfo | None = None
    async with httpx.AsyncClient(
        base_url=settings.select_base_url, timeout=CALL_TIMEOUT_S
    ) as router_http:
        router = RouterClient(router_http, sleep=sleep, clock=clock)
        try:
            statuses = await router.statuses()
            for candidate in candidates:
                if candidate.alias not in statuses:
                    log(f"configuration error: the router does not list preset {candidate.alias}")
                    return EXIT_CONFIG
            for alias, status in statuses.items():
                if status.value != "unloaded" and not await router.unload(alias):
                    log(f"{alias}: the router did not unload a model left from an earlier run")
                    return EXIT_ROUTER
            host = read_host(proc, models_dir, cpus)
            unloaded = await _check_all(
                settings, args, router, candidates, refs, items, results,
                mem_limit=mem_limit, proc=proc, models_dir=models_dir, clock=clock, log=log,
            )  # fmt: skip
        except RouterError as exc:
            log(f"router error: {exc}")
            unloaded = False
    ctx_host = host or read_host(proc, models_dir, cpus)
    ctx = RunContext(ctx_host, settings, now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    if not unloaded:
        if results:
            _emit(args, ctx, candidates, results, refs, out)
        return EXIT_ROUTER
    decision = _emit(args, ctx, candidates, results, refs, out)
    return EXIT_CHOSEN if decision.model is not None else EXIT_NONE


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = SelectSettings()
    except ValidationError as exc:
        names = sorted({".".join(str(part) for part in e["loc"]) for e in exc.errors()})
        print(f"invalid settings: {', '.join(names)}", file=sys.stderr)
        return EXIT_CONFIG
    return asyncio.run(run_selection(settings, args))
