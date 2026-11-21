import argparse
import json
import re
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from docs_helpers import env_example, table_rows

from error_classifier.benchmark.cli import build_parser as bench_parser
from error_classifier.classify.prompt import SYSTEM_PROMPT
from error_classifier.classify.schema import Category, Severity
from error_classifier.config import Settings
from error_classifier.errors import ServiceError
from error_classifier.evaluation.cli import build_parser as eval_parser
from error_classifier.selection.cli import build_parser as selection_parser

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

ENUM_VALUES = {str(v) for v in [*Category, *Severity]}


def _definitions() -> dict[str, str]:
    found: dict[str, str] = {}
    for line in SYSTEM_PROMPT.split("\n"):
        match = re.fullmatch(r"- (\w+): (.+)", line)
        if match and match.group(1) in ENUM_VALUES:
            found[match.group(1)] = match.group(2)
    return found


def test_prompt_definitions_were_parsed() -> None:
    assert set(_definitions()) == ENUM_VALUES


@pytest.mark.parametrize("value", sorted(ENUM_VALUES))
def test_enum_value_row_matches_the_prompt(value: str) -> None:
    assert f"| {value} | {_definitions()[value]} |" in README


def _error_classes() -> list[type[ServiceError]]:
    return list(ServiceError.__subclasses__())


def _coded_subclasses(cls: type[ServiceError]) -> list[type[ServiceError]]:
    found: list[type[ServiceError]] = []
    for sub in cls.__subclasses__():
        if "code" in sub.__dict__:
            found.append(sub)
        found.extend(_coded_subclasses(sub))
    return found


@pytest.mark.parametrize("cls", [ServiceError, *_error_classes()], ids=lambda c: c.__name__)
def test_error_code_row(cls: type[ServiceError]) -> None:
    assert f"| {cls.code} | {cls.status_code} |" in README


def test_every_error_code_is_documented() -> None:
    classes = _coded_subclasses(ServiceError)
    assert classes
    rows = README.splitlines()
    for cls in classes:
        prefix = f"| {cls.code} | {cls.status_code} |"
        assert any(row.startswith(prefix) for row in rows), cls.__name__


def test_compose_variables_match_env_example() -> None:
    compose_text = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    compose_vars = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", compose_text))
    env_keys = set(re.findall(r"^([A-Z_][A-Z0-9_]*)=", env_text, re.M))
    assert compose_vars
    assert compose_vars == env_keys, (
        f"compose only: {sorted(compose_vars - env_keys)}, "
        f".env.example only: {sorted(env_keys - compose_vars)}"
    )


def _api_environment() -> set[str]:
    compose_text = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    start = re.search(r"^  api:$", compose_text, re.M)
    assert start is not None
    rest = compose_text[start.end() :]
    end = re.search(r"^  \S", rest, re.M)
    block = rest[: end.start()] if end else rest
    return set(re.findall(r"^\s{6}([A-Z_][A-Z0-9_]*):", block, re.M))


def test_every_setting_reaches_the_api_container() -> None:
    names = {name.upper() for name in Settings.model_fields}
    missing = sorted(names - _api_environment())
    assert not missing, f"not in the api environment of compose.yaml: {missing}"


def test_no_internal_network_claim() -> None:
    assert "internal network" not in README
    assert not re.search(r"(?i)внутренн\w*\s+(\S+-)?сет", README)


def test_readme_explains_language_and_runtime_api() -> None:
    text = " ".join(README.split())
    assert "`summary` и `suggested_action` всегда на английском" in text
    assert "OpenAI-совместим" in text


def test_readme_example_response_has_the_documented_fields() -> None:
    lines = [line for line in README.splitlines() if line.startswith('{"category"')]
    assert len(lines) == 1
    body = json.loads(lines[0])
    assert set(body) == {
        "category",
        "severity",
        "summary",
        "suggested_action",
        "model",
        "latency_ms",
        "truncated",
    }


def test_only_the_readme_is_markdown() -> None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True
        )
    except FileNotFoundError:
        pytest.skip("git is not available")
    if result.returncode != 0:
        pytest.skip(f"not a git repository: {result.stderr.strip()}")
    assert result.stdout.split() == ["README.md"]


def test_no_docs_directory() -> None:
    assert not (ROOT / "docs").exists()


def _sh_commands(script: str) -> list[str]:
    """Commands of `script` from the sh blocks of the README, joined at a trailing backslash."""
    found: list[str] = []
    in_block = False
    pending = ""
    for raw in README.splitlines():
        if re.fullmatch(r"\s*```sh\s*", raw):
            in_block = True
        elif in_block and raw.strip().startswith("```"):
            in_block = False
            pending = ""
        elif in_block:
            line = raw.rstrip()
            if line.endswith("\\"):
                pending += line[:-1] + " "
                continue
            line = pending + line
            pending = ""
            if f"scripts/{script}.py " in line:
                found.append(line)
    return found


@pytest.mark.parametrize(
    ("script", "parser"),
    [("evaluate", eval_parser), ("bench", bench_parser)],
)
def test_readme_commands_parse(script: str, parser: Callable[[], argparse.ArgumentParser]) -> None:
    lines = _sh_commands(script)
    assert lines
    for line in lines:
        tokens = shlex.split(line, comments=True)
        start = next(i for i, t in enumerate(tokens) if t.endswith(f"scripts/{script}.py"))
        args = tokens[start + 1 :]
        try:
            parser().parse_args(args)
        except SystemExit as exc:
            pytest.fail(f"README command does not parse ({exc.code}): {line}")
        for arg in args:
            if "*" in arg:
                assert list(ROOT.glob(arg)), f"no file matches {arg}: {line}"


def _section(heading: str) -> str:
    assert README.count(f"{heading}\n") == 1, heading
    body = README.split(f"{heading}\n")[1]
    end = re.search(r"^#", body, re.MULTILINE)
    return body[: end.start()] if end else body


def test_readme_settings_table_matches_env_example() -> None:
    env = env_example()
    rows = table_rows(_section("### Настройка"))
    seen = 0
    for row in rows:
        names = row[0].split(", ")
        for name in names:
            assert name in env, name
        seen += len(names)
        defaults = row[1].split(", ")
        if len(defaults) == len(names):
            for name, value in zip(names, defaults, strict=True):
                assert env[name] == ("" if value == "пусто" else value), name
    assert seen >= 10


def test_readme_sizing_first_row_is_env_example() -> None:
    env = env_example()
    lines = README.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Сервер | Модель |"))
    table = [line for line in lines[start:] if line.startswith("|")]
    header = [cell.strip() for cell in table[0].strip("|").split("|")]
    first = [cell.strip() for cell in table[2].strip("|").split("|")]
    checked: list[str] = []
    for head, cell in zip(header[2:], first[2:], strict=True):
        names = re.findall(r"[A-Z_][A-Z0-9_]*", head)
        values = cell.split(" / ")
        assert len(names) == len(values), head
        for name, value in zip(names, values, strict=True):
            assert env[name] == value, name
            checked.append(name)
    assert checked == [
        "LLM_CPUS",
        "LLAMA_ARG_THREADS",
        "LLAMA_ARG_N_PARALLEL",
        "LLAMA_ARG_CTX_SIZE",
        "LLM_MEM_LIMIT",
        "REQUEST_TIMEOUT_S",
        "QUEUE_TIMEOUT_S",
    ]


def test_readme_stays_short() -> None:
    assert len(README) <= 17000


def test_readme_selection_command() -> None:
    lines: list[str] = []
    in_block = False
    for raw in README.splitlines():
        if re.fullmatch(r"\s*```sh\s*", raw):
            in_block = True
        elif in_block and raw.strip().startswith("```"):
            in_block = False
        elif in_block and "--profile select run" in raw and raw.rstrip().find("selector") > 0:
            lines.append(raw)
    assert len(lines) == 1
    assert "docker compose --profile select stop" in README.split("--profile select run")[0]
    tokens = shlex.split(lines[0], comments=True)
    assert tokens[:6] == ["docker", "compose", "--profile", "select", "run", "--rm"]
    assert "--build" in tokens
    assert "-T" in tokens
    rest = tokens[tokens.index("selector") + 1 :]
    if ">" in rest:
        rest = rest[: rest.index(">")]
    try:
        selection_parser().parse_args(rest)
    except SystemExit as exc:
        pytest.fail(f"README command does not parse ({exc.code}): {lines[0]}")


def test_env_example_keeps_names_with_digits() -> None:
    assert "SELECT_TARGET_P95_S" in env_example()
