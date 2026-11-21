import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from bench_files import GIB, bench_file

from error_classifier.benchmark.report import write_bench
from error_classifier.classify.prompt import INPUT_PREFIX
from error_classifier.classify.schema import Category
from error_classifier.evaluation.dataset import DEFAULT_DATASET, EvalItem, load_dataset
from error_classifier.evaluation.metrics import AttemptRecord, ItemResult, summarize
from error_classifier.evaluation.results import (
    ResultDoc,
    RunMeta,
    RunSettings,
    dataset_sha256,
    prompt_sha256,
    write_result,
)
from error_classifier.selection.cli import build_parser, main, run_selection
from error_classifier.selection.config import SelectSettings

pytestmark = pytest.mark.anyio

ROUTER = "http://llm-select:8080"
MAIN = "http://llm:8080"
REPO_ROOT = Path(__file__).parent.parent
INI = """\
[alpha]
hf-repo = org/alpha
hf-file = alpha.gguf

[beta]
hf-repo = org/beta
hf-file = beta.gguf
"""


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Server:
    """Fake of the router and the runtime behind it, driven by the requests it gets."""

    clock: Clock
    secrets: list[str]
    status: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    posts: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    requests: list[str] = field(default_factory=list[str])
    load_fail: dict[str, int] = field(default_factory=dict[str, int])
    stuck_unload: set[str] = field(default_factory=set[str])
    chat_calls: dict[str, int] = field(default_factory=dict[str, int])
    chat_errors: dict[tuple[str, int], httpx.Response | Exception] = field(
        default_factory=dict[tuple[str, int], httpx.Response | Exception]
    )
    model_override: dict[str, str] = field(default_factory=dict[str, str])
    leak: set[str] = field(default_factory=set[str])
    step_s: dict[str, float] = field(default_factory=dict[str, float])
    down: bool = False

    def models(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        if self.down:
            raise httpx.ConnectError("router is down")
        data = [{"id": a, "source": "preset", "status": s} for a, s in self.status.items()]
        return httpx.Response(200, json={"data": data})

    def load(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        alias = json.loads(request.content)["model"]
        self.posts.append(("load", alias))
        if alias in self.load_fail:
            code = self.load_fail[alias]
            self.status[alias] = {"value": "unloaded", "failed": True, "exit_code": code}
        else:
            self.status[alias] = {"value": "loaded"}
        return httpx.Response(200, json={"success": True})

    def unload(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        alias = json.loads(request.content)["model"]
        self.posts.append(("unload", alias))
        if alias not in self.stuck_unload:
            self.status[alias] = {"value": "unloaded"}
        return httpx.Response(200, json={"success": True})

    def chat(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        body = json.loads(request.content)
        alias = body["model"]
        last = next(
            m["content"]
            for m in reversed(body["messages"])
            if m["content"].startswith(INPUT_PREFIX)
        )
        text = str(json.loads(last.removeprefix(INPUT_PREFIX)))
        number = self.chat_calls[alias] = self.chat_calls.get(alias, 0) + 1
        self.clock.now += self.step_s.get(alias, 1.0)
        error = self.chat_errors.get((alias, number))
        if isinstance(error, Exception):
            raise error
        if error is not None:
            return error
        summary = "It failed."
        found = [s for s in self.secrets if s in text]
        if alias in self.leak and found:
            summary = f"The failure was {found[0]}."
        answer = {
            "category": "runtime_error",
            "severity": "medium",
            "summary": summary,
            "suggested_action": "Check the code.",
        }
        reply = {
            "model": self.model_override.get(alias, alias),
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(200, json=reply)


@dataclass
class World:
    root: Path
    server: Server
    clock: Clock
    mock: respx.MockRouter
    items: list[EvalItem]

    @property
    def settings(self) -> SelectSettings:
        return SelectSettings(
            presets_path=self.root / "presets.ini",
            data_root=self.root,
            select_model_max_s=1200.0,
        )

    def reference(
        self, alias: str, *, leak: bool = False, wrong: int = 0, prompt: str | None = None
    ) -> None:
        results: list[ItemResult] = []
        leaked = False
        for i, item in enumerate(self.items):
            category = item.category
            if i < wrong:
                category = (
                    Category.UNKNOWN if item.category != Category.UNKNOWN else Category.SYNTAX_ERROR
                )
            summary = "It failed."
            if leak and not leaked and item.secret is not None:
                summary, leaked = f"It failed with {item.secret}.", True
            results.append(
                ItemResult(
                    id=item.id,
                    outcome="ok",
                    attempts=1,
                    latency_ms=1000,
                    truncated=False,
                    prompt_tokens=10,
                    completion_tokens=5,
                    category=category,
                    severity=item.severity,
                    summary=summary,
                    suggested_action="Check.",
                    attempt_log=[
                        AttemptRecord(
                            content="{}",
                            finish_reason="stop",
                            prompt_tokens=10,
                            completion_tokens=5,
                            error=None,
                        )
                    ],
                )
            )
        meta = RunMeta(
            model=alias, engine="podman", container="c", runtime_env={}, runtime_build=None,
            model_path=None, total_slots=None, slot_ctx=None,
            settings=RunSettings(
                temperature=0.0, max_output_tokens=256, max_input_chars=3000,
                request_timeout_s=120.0, disable_thinking=False,
            ),
            dataset_sha256=dataset_sha256(self.root / DEFAULT_DATASET),
            prompt_sha256=prompt or prompt_sha256(),
            git_sha=None, git_dirty=None, started_at="2026-11-14T10:00:00Z", wall_s=1.0,
            loadavg_before=None, loadavg_after=None, complete=True, stop_reason=None,
            cpu_count=8, machine="x86_64",
        )  # fmt: skip
        doc = ResultDoc(format=1, meta=meta, summary=summarize(self.items, results), items=results)
        write_result(self.root / "eval" / "results" / f"{alias}.json", doc)

    def drop_references(self) -> None:
        for path in (self.root / "eval" / "results").glob("*.json"):
            path.unlink()

    async def run(self, *argv: str, settings: SelectSettings | None = None) -> tuple[int, str, str]:
        out, err = StringIO(), StringIO()
        code = await run_selection(
            settings or self.settings,
            build_parser().parse_args(list(argv)),
            proc=self.root / "proc",
            models_dir=self.root / "models",
            cpus=lambda: 4,
            sleep=self.clock.sleep,
            clock=self.clock,
            now=lambda: datetime(2026, 11, 21, 10, 0, tzinfo=UTC),
            out=out,
            err=err,
        )
        return code, out.getvalue(), err.getvalue()


@pytest.fixture
def world(tmp_path: Path) -> Iterator[World]:
    (tmp_path / "eval" / "results").mkdir(parents=True)
    shutil.copy(REPO_ROOT / DEFAULT_DATASET, tmp_path / DEFAULT_DATASET)
    (tmp_path / "presets.ini").write_text(INI, encoding="utf-8")
    items = load_dataset(tmp_path / DEFAULT_DATASET)
    clock = Clock()
    server = Server(clock, [i.secret for i in items if i.secret is not None])
    server.status = {a: {"value": "unloaded"} for a in ("alpha", "beta")}
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{ROUTER}/models").mock(side_effect=server.models)
        mock.post(f"{ROUTER}/models/load").mock(side_effect=server.load)
        mock.post(f"{ROUTER}/models/unload").mock(side_effect=server.unload)
        mock.post(f"{ROUTER}/v1/chat/completions").mock(side_effect=server.chat)
        mock.get(f"{MAIN}/health").mock(side_effect=httpx.ConnectError("no main llm"))
        w = World(tmp_path, server, clock, mock, items)
        w.reference("alpha")
        w.reference("beta", wrong=20)
        yield w


async def test_cli_chooses_and_prints_env(world: World) -> None:
    code, out, err = await world.run()
    assert code == 0
    assert "Chosen: alpha" in out
    assert "MODEL_ALIAS=alpha" in out
    assert "MODEL_ALIAS=beta" not in out
    assert world.server.posts == [
        ("load", "alpha"), ("unload", "alpha"), ("load", "beta"), ("unload", "beta"),
    ]  # fmt: skip
    assert "alpha: loaded in" in err
    assert "alpha: row 1/12 ok" in err
    assert "error" not in err.lower().replace("error text", "")


async def test_cli_reference_leak_not_chosen(world: World) -> None:
    world.reference("beta", leak=True)
    world.server.leak = {"alpha"}
    code, out, _ = await world.run()
    assert code == 3
    assert "reference failed thresholds" in out
    assert "No model chosen:" in out


async def test_cli_smoke_leak_fails(world: World) -> None:
    world.server.leak = {"alpha"}
    code, out, _ = await world.run()
    assert code == 0
    assert "leaked" in out
    assert "Chosen: beta" in out


async def test_cli_runtime_stop_fails(world: World) -> None:
    world.server.chat_errors = {("alpha", 4): httpx.Response(503)}
    code, out, _ = await world.run()
    assert code == 0
    assert "smoke stopped after 2 of 12 rows (llm_unavailable)" in out
    assert "Chosen: beta" in out


async def test_cli_other_model_fails(world: World) -> None:
    world.server.model_override = {"alpha": "other"}
    _, out, _ = await world.run()
    assert "answers came from another model" in out
    assert "Chosen: beta" in out


async def test_cli_load_failed_137(world: World) -> None:
    world.server.load_fail = {"alpha": 137}
    _, out, _ = await world.run()
    assert "failed to load (exit code 137)" in out
    assert "Chosen: beta" in out


async def test_cli_unloads_after_a_failed_load(world: World) -> None:
    world.server.load_fail = {"alpha": 137}
    await world.run()
    assert world.server.posts[:3] == [("load", "alpha"), ("unload", "alpha"), ("load", "beta")]


async def test_cli_stale_reference(world: World) -> None:
    world.reference("alpha", prompt="0" * 64)
    _, out, _ = await world.run()
    assert "reference is stale (prompt changed)" in out
    assert "Chosen: beta" in out


async def test_cli_no_references_picks_fastest(world: World) -> None:
    world.drop_references()
    world.server.step_s = {"alpha": 3.0, "beta": 1.0}
    code, out, _ = await world.run()
    assert code == 0
    assert "Chosen: beta" in out
    assert "no reference quality" in out


async def test_cli_main_llm_running_exits_2(world: World) -> None:
    world.mock.get(f"{MAIN}/health").mock(return_value=httpx.Response(503))
    code, out, err = await world.run()
    assert code == 2
    assert "stop the main stack first: docker compose stop" in err
    assert out == ""
    assert world.server.requests == []


async def test_cli_main_llm_running_allowed_with_flag(world: World) -> None:
    world.mock.get(f"{MAIN}/health").mock(return_value=httpx.Response(200))
    code, _, _ = await world.run("--allow-running")
    assert code == 0


async def test_cli_main_llm_absent_runs(world: World) -> None:
    code, _, _ = await world.run()
    assert code == 0
    assert world.server.requests != []


async def test_cli_unloads_leftover_first(world: World) -> None:
    world.server.status["beta"] = {"value": "loaded"}
    code, _, _ = await world.run()
    assert code == 0
    assert world.server.posts[0] == ("unload", "beta")


@pytest.mark.parametrize("value", ["loading", "downloading"])
async def test_cli_unloads_an_interrupted_load_first(world: World, value: str) -> None:
    world.server.status["beta"] = {"value": value}
    code, _, _ = await world.run()
    assert code == 0
    assert world.server.posts[0] == ("unload", "beta")


async def test_cli_warmup_error_fails_candidate(world: World) -> None:
    world.server.chat_errors = {("alpha", 1): httpx.ReadTimeout("slow")}
    _, out, _ = await world.run()
    assert "warm-up failed: llm_timeout" in out
    assert "Chosen: beta" in out
    assert world.server.posts == [
        ("load", "alpha"), ("unload", "alpha"), ("load", "beta"), ("unload", "beta"),
    ]  # fmt: skip


async def test_cli_unload_timeout_exits_4(world: World) -> None:
    world.server.stuck_unload = {"alpha"}
    code, out, err = await world.run()
    assert code == 4
    assert "# Model selection" in out
    assert "alpha" in err
    assert world.server.posts == [("load", "alpha"), ("unload", "alpha")]


async def test_cli_stuck_leftover_exits_4_without_a_report(world: World) -> None:
    world.server.status["alpha"] = {"value": "loaded"}
    world.server.stuck_unload = {"alpha"}
    code, out, _ = await world.run()
    assert code == 4
    assert out == ""


async def test_cli_router_down_exits_4(world: World) -> None:
    world.server.down = True
    code, out, err = await world.run()
    assert code == 4
    assert out == ""
    assert "router" in err


async def test_cli_bad_presets_exits_2(world: World) -> None:
    (world.root / "presets.ini").write_text("[alpha]\nhf-repo = org/x\n", encoding="utf-8")
    code, _, err = await world.run()
    assert code == 2
    assert "[alpha]" in err
    assert world.server.requests == []


async def test_cli_unknown_preset_exits_2(world: World) -> None:
    del world.server.status["beta"]
    code, _, err = await world.run()
    assert code == 2
    assert "the router does not list preset beta" in err
    assert world.server.posts == []


async def test_cli_json(world: World) -> None:
    code, out, _ = await world.run("--json")
    assert code == 0
    assert json.loads(out)["decision"]["model"] == "alpha"


async def test_cli_prefilter_skips_and_all_checks(world: World) -> None:
    big = bench_file("beta", conf={"model_file_size": 5 * GIB})
    write_bench(
        world.root / "bench" / "results" / "2026-11-15-beta.jsonl",
        big.records, big.readings, big.summary,
    )  # fmt: skip
    _, out, _ = await world.run()
    assert ("load", "beta") not in world.server.posts
    assert "reference needs 7.00 GiB, above LLM_MEM_LIMIT" in out
    world.server.posts.clear()
    await world.run("--all")
    assert ("load", "beta") in world.server.posts


def test_main_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_main_bad_settings_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SELECT_TARGET_P95_S", "-5")
    assert main([]) == 2
    err = capsys.readouterr().err
    assert "select_target_p95_s" in err
    assert "-5" not in err


def test_select_settings_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SelectSettings()
    assert settings.select_base_url == "http://llm-select:8080"
    assert settings.llm_mem_limit == "6656m"
    assert settings.select_model_max_s == 1200.0
    monkeypatch.setenv("SELECT_BASE_URL", "http://x:1")
    monkeypatch.setenv("LLAMA_ARG_N_PARALLEL", "4")
    monkeypatch.setenv("UNRELATED", "1")
    changed = SelectSettings()
    assert changed.select_base_url == "http://x:1"
    assert changed.llama_arg_n_parallel == 4
