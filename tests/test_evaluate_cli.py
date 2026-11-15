import json
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
import respx

from error_classifier.classify.prompt import INPUT_PREFIX
from error_classifier.evaluation import cli, results
from error_classifier.evaluation.dataset import DEFAULT_DATASET, EvalItem, load_dataset
from error_classifier.evaluation.results import (
    ResultDoc,
    container_env,
    dataset_sha256,
    prompt_sha256,
    read_result,
    render_report,
)

ROOT = Path(__file__).resolve().parents[1]
ENV = {"LLAMA_ARG_THREADS": "6", "LLAMA_ARG_ALIAS": "m-test"}
PROPS = {
    "build_info": "b12352-test",
    "model_path": "/models/m.gguf",
    "total_slots": 2,
    "default_generation_settings": {"n_ctx": 4096},
}
ITEMS = load_dataset(ROOT / DEFAULT_DATASET)
BY_TEXT = {item.text.strip(): item for item in ITEMS}


class Fake:
    """Fake llama-server routes: gold answers, an invalid first answer for js-001."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.calls: dict[str, int] = {}
        self.models = ["m-test", "m-other"]
        router.get("/v1/models").mock(side_effect=self._models)
        router.get("/props").mock(return_value=httpx.Response(200, json=PROPS))
        router.post("/v1/chat/completions").mock(side_effect=self._chat)
        self.router = router

    def _models(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})

    def _chat(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        last = next(
            m["content"]
            for m in reversed(body["messages"])
            if m["content"].startswith(INPUT_PREFIX)
        )
        item: EvalItem | None = BY_TEXT.get(json.loads(last.removeprefix(INPUT_PREFIX)))
        category, severity = (item.category, item.severity) if item else ("unknown", "low")
        key = item.id if item else "warmup"
        self.calls[key] = self.calls.get(key, 0) + 1
        if key == "js-001" and self.calls[key] == 1:
            content = "not json"
        else:
            content = json.dumps(
                {
                    "category": str(category),
                    "severity": str(severity),
                    "summary": "The code failed.",
                    "suggested_action": "Check the code.",
                }
            )
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )


def _env_of_container(engine: str, container: str) -> dict[str, str] | None:
    return dict(ENV)


def _no_env(engine: str, container: str) -> dict[str, str] | None:
    return None


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[Fake]:
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "m-test")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(results, "container_env", _env_of_container)
    with respx.mock(base_url="http://llm.test", assert_all_called=False) as router:
        yield Fake(router)


class Env:
    def __init__(self, loads: list[float]) -> None:
        self.loads = loads
        self.pauses: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.pauses.append(seconds)

    def loadavg(self) -> tuple[float, float, float]:
        value = self.loads.pop(0) if len(self.loads) > 1 else self.loads[0]
        return (value, value, value)


def _run(
    out: Path,
    *extra: str,
    loads: list[float] | None = None,
    dataset: Path | None = None,
) -> tuple[int, Env]:
    env = Env(loads or [0.5])
    args = ["run", "--out", str(out), "--engine", "podman", "--container", "llm-1", *extra]
    if dataset is not None:
        args += ["--dataset", str(dataset)]
    code = cli.main(args, sleep=env.sleep, loadavg=env.loadavg)
    return code, env


def test_run_writes_complete_result(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "res" / "m-test.json"
    code, _ = _run(out)
    assert code == 0
    doc = read_result(out)
    meta = doc.meta
    assert meta.complete is True
    assert len(doc.items) == 60
    assert (meta.model, meta.engine, meta.container) == ("m-test", "podman", "llm-1")
    assert meta.runtime_env == ENV
    assert meta.runtime_build == "b12352-test"
    assert meta.model_path == "/models/m.gguf"
    assert (meta.total_slots, meta.slot_ctx) == (2, 4096)
    assert meta.dataset_sha256 == dataset_sha256(ROOT / DEFAULT_DATASET)
    assert meta.prompt_sha256 == prompt_sha256()
    assert meta.settings.max_output_tokens == 256
    assert meta.loadavg_before == [0.5, 0.5, 0.5]
    assert meta.loadavg_after == [0.5, 0.5, 0.5]
    js1 = next(r for r in doc.items if r.id == "js-001")
    assert js1.attempts == 2
    assert len(js1.attempt_log) == 2
    assert doc.summary.both_correct == 60
    assert "js-001 ok attempts=2" in capsys.readouterr().err


def test_run_without_runtime_exits_3_and_writes_nothing(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.router.get("/v1/models").mock(side_effect=httpx.ConnectError("refused"))
    out = tmp_path / "m.json"
    code, _ = _run(out)
    assert code == 3
    assert "LLM runtime is not ready" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_run_with_runtime_dying_in_warmup_exits_3(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.router.post("/v1/chat/completions").mock(return_value=httpx.Response(503))
    code, _ = _run(tmp_path / "m.json")
    assert code == 3
    assert "warm-up failed: llm_unavailable" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_run_with_invalid_warmup_answers_exits_3(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answer = {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]}
    fake.router.post("/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "m-test", **answer})
    )
    code, _ = _run(tmp_path / "m.json")
    assert code == 3
    assert "warm-up failed: model_output_invalid" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_run_with_broken_props_still_runs(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.router.get("/props").mock(return_value=httpx.Response(404))
    out = tmp_path / "m.json"
    code, _ = _run(out)
    assert code == 0
    meta = read_result(out).meta
    assert (meta.runtime_build, meta.model_path, meta.total_slots, meta.slot_ctx) == (
        None,
        None,
        None,
        None,
    )
    assert "warning: /props unavailable, runtime fields left empty" in capsys.readouterr().err


def test_props_with_wrong_types_gives_none(fake: Fake, tmp_path: Path) -> None:
    bad = {
        "build_info": 5,
        "model_path": ["x"],
        "total_slots": True,
        "default_generation_settings": {"n_ctx": "4096"},
    }
    fake.router.get("/props").mock(return_value=httpx.Response(200, json=bad))
    out = tmp_path / "m.json"
    assert _run(out)[0] == 0
    meta = read_result(out).meta
    assert (meta.runtime_build, meta.model_path, meta.total_slots, meta.slot_ctx) == (
        None,
        None,
        None,
        None,
    )


def test_unreadable_container_env_warns_and_leaves_it_empty(
    fake: Fake,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(results, "container_env", _no_env)
    out = tmp_path / "m.json"
    assert _run(out)[0] == 0
    assert read_result(out).meta.runtime_env == {}
    assert "warning: cannot read the env of container llm-1" in capsys.readouterr().err


def test_limit_writes_partial_only(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.json"
    code, _ = _run(out, "--limit", "3")
    assert code == 4
    assert not out.exists()
    partial = read_result(tmp_path / "m.json.partial")
    assert len(partial.items) == 3
    assert partial.meta.complete is False
    assert partial.meta.stop_reason == "limit"
    assert "run stopped: limit" in capsys.readouterr().err


def test_limit_above_set_size_is_still_partial(fake: Fake, tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    code, _ = _run(out, "--limit", "100")
    assert code == 4
    assert not out.exists()
    partial = read_result(tmp_path / "m.json.partial")
    assert len(partial.items) == 60
    assert partial.meta.complete is False


def test_limit_must_be_positive(fake: Fake, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stop:
        _run(tmp_path / "m.json", "--limit", "0")
    assert stop.value.code == 2


def test_partial_does_not_replace_complete(fake: Fake, tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    assert _run(out)[0] == 0
    before = out.read_bytes()
    assert _run(out, "--limit", "3")[0] == 4
    assert out.read_bytes() == before
    assert (tmp_path / "m.json.partial").exists()


def test_run_rejects_bad_dataset(fake: Fake, tmp_path: Path) -> None:
    short = tmp_path / "short.jsonl"
    first = (ROOT / DEFAULT_DATASET).read_text(encoding="utf-8").splitlines()[0]
    short.write_text(first + "\n", encoding="utf-8")
    out = tmp_path / "out" / "m.json"
    code, _ = _run(out, dataset=short)
    assert code == 2
    assert not out.parent.exists()


def test_run_rejects_broken_dataset_line(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n", encoding="utf-8")
    code, _ = _run(tmp_path / "m.json", dataset=broken)
    assert code == 2
    assert "line 1" in capsys.readouterr().err


def test_run_rejects_bad_settings(
    fake: Fake,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("LLM_MODEL")
    code, _ = _run(tmp_path / "m.json")
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("settings: ")
    assert "llm_model" in err
    assert list(tmp_path.iterdir()) == []


def test_run_waits_for_load(fake: Fake, tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    code, env = _run(out, loads=[3.0, 2.0, 1.0])
    assert code == 0
    assert env.pauses == [15, 15]
    assert read_result(out).meta.loadavg_before == [1.0, 1.0, 1.0]


def test_run_stops_waiting_for_load_after_ten_minutes(fake: Fake, tmp_path: Path) -> None:
    code, env = _run(tmp_path / "m.json", "--limit", "1", loads=[3.0])
    assert code == 4
    assert env.pauses == [15] * 40


def test_container_env_parses_inspect_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(code: int, out: str) -> Callable[..., subprocess.CompletedProcess[str]]:
        def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            assert cmd[:4] == ["podman", "inspect", "--format", "{{json .Config.Env}}"]
            return subprocess.CompletedProcess(cmd, code, out, "")

        return run

    listing = '["PATH=/bin","LLAMA_ARG_THREADS=6","LLAMA_ARG_HF_FILE=a=b.gguf","NOEQUALS"]'
    monkeypatch.setattr(subprocess, "run", fake_run(0, listing))
    assert container_env("podman", "llm-1") == {
        "LLAMA_ARG_THREADS": "6",
        "LLAMA_ARG_HF_FILE": "a=b.gguf",
    }
    monkeypatch.setattr(subprocess, "run", fake_run(1, listing))
    assert container_env("podman", "llm-1") is None
    monkeypatch.setattr(subprocess, "run", fake_run(0, "null"))
    assert container_env("podman", "llm-1") is None


def _two_results(fake: Fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    first, second = tmp_path / "m-test.json", tmp_path / "m-other.json"
    assert _run(first)[0] == 0
    monkeypatch.setenv("LLM_MODEL", "m-other")
    assert _run(second)[0] == 0
    return first, second


def test_report_prints_tables(
    fake: Fake,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first, second = _two_results(fake, tmp_path, monkeypatch)
    capsys.readouterr()
    assert cli.main(["report", str(first), str(second)]) == 0
    docs: list[ResultDoc] = [read_result(first), read_result(second)]
    expected = render_report(docs, ITEMS)
    out = capsys.readouterr().out
    assert out == expected
    assert "Choice by the rule" in out


def test_report_rejects_partial(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(tmp_path / "m.json", "--limit", "3")
    capsys.readouterr()
    assert cli.main(["report", str(tmp_path / "m.json.partial")]) == 2
    assert "incomplete" in capsys.readouterr().err


def test_report_rejects_a_broken_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "x.json"
    broken.write_text("{}", encoding="utf-8")
    assert cli.main(["report", str(broken)]) == 2
    assert str(broken) in capsys.readouterr().err


def test_report_rejects_a_foreign_dataset(
    fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "m.json"
    assert _run(out)[0] == 0
    other = tmp_path / "other.jsonl"
    text = (ROOT / DEFAULT_DATASET).read_text(encoding="utf-8")
    other.write_text(text.replace('"note": "', '"note": "x ', 1), encoding="utf-8")
    capsys.readouterr()
    assert cli.main(["report", "--dataset", str(other), str(out)]) == 2
    assert "dataset" in capsys.readouterr().err


def test_report_warns_about_another_prompt(
    fake: Fake,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "m.json"
    assert _run(out)[0] == 0
    monkeypatch.setattr(results, "prompt_sha256", lambda: "0" * 64)
    capsys.readouterr()
    assert cli.main(["report", str(out)]) == 0
    assert "warning: m-test was run with another prompt" in capsys.readouterr().err
