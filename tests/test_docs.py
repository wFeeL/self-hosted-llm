import json
import re
import subprocess
from pathlib import Path

import pytest

from error_classifier.classify.prompt import SYSTEM_PROMPT
from error_classifier.classify.schema import Category, Severity
from error_classifier.errors import ServiceError

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

ENUM_VALUES = {str(v) for v in [*Category, *Severity]}
SETTINGS_VARIABLES = (
    "REQUEST_TIMEOUT_S",
    "MAX_CONCURRENCY",
    "MAX_QUEUE",
    "QUEUE_TIMEOUT_S",
    "MAX_INPUT_CHARS",
    "MAX_REQUEST_BYTES",
    "MAX_OUTPUT_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_DISABLE_THINKING",
)


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


@pytest.mark.parametrize("cls", [ServiceError, *_error_classes()], ids=lambda c: c.__name__)
def test_error_code_row(cls: type[ServiceError]) -> None:
    assert f"| {cls.code} | {cls.status_code} |" in README


@pytest.mark.parametrize("name", SETTINGS_VARIABLES)
def test_setting_is_documented(name: str) -> None:
    assert f"| {name} |" in README


def test_no_internal_network_claim() -> None:
    assert "internal network" not in README


def test_readme_explains_language_and_runtime_api() -> None:
    assert "English" in README
    assert "OpenAI-compatible" in README


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
