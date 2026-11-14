import argparse
import asyncio
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import httpx
from pydantic import ValidationError

from error_classifier.classify.service import ClassifierOptions
from error_classifier.config import Settings, load_settings
from error_classifier.errors import ServiceError
from error_classifier.evaluation import results
from error_classifier.evaluation.dataset import (
    DEFAULT_DATASET,
    DatasetError,
    EvalItem,
    check_dataset,
    few_shot_texts,
    load_dataset,
)
from error_classifier.evaluation.metrics import LOAD_LIMIT, ItemResult, summarize
from error_classifier.evaluation.results import ResultDoc, RunMeta, RunSettings
from error_classifier.evaluation.runner import ENVIRONMENT_ERRORS, run_eval
from error_classifier.llm.client import LlmClient, build_http_client

LOAD_WAIT_MAX_S = 600
LOAD_WAIT_STEP_S = 15

type LoadAvg = Callable[[], tuple[float, float, float]]


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the set through the classifier")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--engine", choices=("podman", "docker"), required=True)
    run.add_argument("--container", required=True)
    run.add_argument("--max-minutes", type=float, default=15.0)
    run.add_argument("--limit", type=_positive, default=None)
    report = commands.add_parser("report", help="print tables for result files")
    report.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    report.add_argument("paths", type=Path, nargs="+", metavar="PATH")
    return parser


def _str_field(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _int_field(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    # bool is an int subclass, but a flag is not a count.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


async def _props(http: httpx.AsyncClient, settings: Settings) -> dict[str, object] | None:
    url = settings.llm_base_url.removesuffix("/v1") + "/props"
    try:
        response = await http.get(url, timeout=settings.llm_probe_timeout_s)
        body = cast(object, response.json()) if response.status_code == 200 else None
    except httpx.HTTPError, ValueError:
        body = None
    if not isinstance(body, dict):
        _err("warning: /props unavailable, runtime fields left empty")
        return None
    return cast(dict[str, object], body)


def _wait_for_load(sleep: Callable[[float], None], loadavg: LoadAvg) -> tuple[float, float, float]:
    current = loadavg()
    waited = 0
    while current[0] >= LOAD_LIMIT and waited < LOAD_WAIT_MAX_S:
        _err("waiting for load ...")
        sleep(LOAD_WAIT_STEP_S)
        waited += LOAD_WAIT_STEP_S
        current = loadavg()
    return current


def _progress(r: ItemResult) -> None:
    _err(f"{r.id} {r.outcome} attempts={r.attempts} {r.latency_ms / 1000:.1f}s")


async def _run_async(
    args: argparse.Namespace,
    settings: Settings,
    items: list[EvalItem],
    sleep: Callable[[float], None],
    loadavg: LoadAvg,
) -> int:
    out = cast(Path, args.out)
    async with build_http_client(settings) as http:
        llm = LlmClient(http, settings.llm_model, settings.llm_probe_timeout_s)
        try:
            await llm.check_ready()
        except ServiceError as exc:
            _err(f"LLM runtime is not ready: {exc}")
            return 3
        props = await _props(http, settings)
        env = results.container_env(cast(str, args.engine), cast(str, args.container))
        if env is None:
            _err(f"warning: cannot read the env of container {args.container}")
            env = {}
        before = _wait_for_load(sleep, loadavg)
        started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        git_sha, git_dirty = results.git_state(Path.cwd())
        try:
            outcome = await run_eval(
                items,
                llm,
                ClassifierOptions.from_settings(settings),
                max_s=cast(float, args.max_minutes) * 60,
                limit=cast("int | None", args.limit),
                probe=llm.check_ready,
                on_item=_progress,
            )
        except ENVIRONMENT_ERRORS as exc:
            _err(f"LLM runtime is not ready: {exc}")
            return 3

    after = loadavg()
    props = props or {}
    slot_settings = props.get("default_generation_settings")
    slot_ctx = (
        _int_field(cast(dict[str, object], slot_settings), "n_ctx")
        if isinstance(slot_settings, dict)
        else None
    )
    done = outcome.results
    complete = outcome.stop_reason is None and args.limit is None and len(done) == len(items)
    doc = ResultDoc(
        format=1,
        meta=RunMeta(
            model=settings.llm_model,
            engine=cast(Literal["podman", "docker"], args.engine),
            container=cast(str, args.container),
            runtime_env=env,
            runtime_build=_str_field(props, "build_info"),
            model_path=_str_field(props, "model_path"),
            total_slots=_int_field(props, "total_slots"),
            slot_ctx=slot_ctx,
            settings=RunSettings(
                temperature=settings.llm_temperature,
                max_output_tokens=settings.max_output_tokens,
                max_input_chars=settings.max_input_chars,
                request_timeout_s=settings.request_timeout_s,
                disable_thinking=settings.llm_disable_thinking,
            ),
            dataset_sha256=results.dataset_sha256(cast(Path, args.dataset)),
            prompt_sha256=results.prompt_sha256(),
            git_sha=git_sha,
            git_dirty=git_dirty,
            started_at=started_at,
            wall_s=round(outcome.wall_s, 1),
            loadavg_before=[round(x, 2) for x in before],
            loadavg_after=[round(x, 2) for x in after],
            complete=complete,
            stop_reason=outcome.stop_reason,
            cpu_count=os.cpu_count(),
            machine=platform.machine(),
        ),
        summary=summarize(items[: len(done)], done),
        items=done,
    )
    target = out if complete else out.with_name(out.name + ".partial")
    results.write_result(target, doc)
    if complete:
        return 0
    _err(f"run stopped: {outcome.stop_reason or 'limit'}, wrote {target}")
    return 4


def _run(args: argparse.Namespace, sleep: Callable[[float], None], loadavg: LoadAvg) -> int:
    try:
        settings = load_settings()
    except ValidationError as exc:
        names = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
        _err(f"settings: {names}")
        return 2
    try:
        items = load_dataset(cast(Path, args.dataset))
    except (DatasetError, OSError) as exc:
        _err(str(exc))
        return 2
    problems = check_dataset(items, few_shot_texts(), settings.max_input_chars)
    if problems:
        for problem in problems:
            _err(problem)
        return 2
    return asyncio.run(_run_async(args, settings, items, sleep, loadavg))


def _report(args: argparse.Namespace) -> int:
    dataset = cast(Path, args.dataset)
    try:
        items = load_dataset(dataset)
        dataset_hash = results.dataset_sha256(dataset)
    except (DatasetError, OSError) as exc:
        _err(str(exc))
        return 2
    docs: list[ResultDoc] = []
    for path in cast("list[Path]", args.paths):
        try:
            doc = results.read_result(path)
        except (OSError, ValueError) as exc:
            _err(f"{path}: {exc}")
            return 2
        if doc.meta.dataset_sha256 != dataset_hash:
            _err(f"{path}: the result was made on a different dataset than {dataset}")
            return 2
        docs.append(doc)
    try:
        text = results.render_report(docs, items)
    except ValueError as exc:
        _err(str(exc))
        return 2
    sys.stdout.write(text)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    loadavg: LoadAvg = os.getloadavg,
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args, sleep, loadavg)
    return _report(args)
