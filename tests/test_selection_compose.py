import re

from docs_helpers import ROOT

COMPOSE = (ROOT / "compose.yaml").read_text(encoding="utf-8")


def service_block(name: str) -> str:
    lines = COMPOSE.splitlines()
    start = lines.index(f"  {name}:")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.startswith("volumes:") or re.fullmatch(r"  [\w-]+:", line):
            end = i
            break
    return "\n".join(lines[start:end])


def _command(block: str) -> list[str]:
    body = block.split("    command:\n")[1]
    items: list[str] = []
    for line in body.splitlines():
        match = re.fullmatch(r'      - "(.*)"', line)
        if match is None:
            break
        items.append(match.group(1))
    return items


def _lines(block: str, key: str) -> list[str]:
    return [line.strip() for line in block.splitlines() if line.startswith(f"    {key}:")]


def test_llm_select_profile_and_args() -> None:
    block = service_block("llm-select")
    assert '    profiles: ["select"]' in block
    command = _command(block)
    expected = [
        "--models-preset",
        "/select/presets.ini",
        "--models-max",
        "1",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--ctx-size",
        "${LLAMA_ARG_CTX_SIZE:-8192}",
        "--parallel",
        "${LLAMA_ARG_N_PARALLEL:-2}",
        "--threads",
        "${LLAMA_ARG_THREADS:-6}",
        "--batch-size",
        "${LLAMA_ARG_BATCH:-512}",
        "--cache-ram",
        "${LLAMA_ARG_CACHE_RAM:-0}",
        "--no-mmproj-auto",
        "--no-models-autoload",
    ]
    assert command == expected


def test_llm_select_env_only_cache() -> None:
    block = service_block("llm-select")
    env = block.split("    environment:\n")[1].split("\n    volumes:")[0]
    keys = re.findall(r"^      ([A-Z_]+):", env, re.M)
    assert keys == ["LLAMA_CACHE"]
    assert "LLAMA_ARG_" not in env


def test_llm_select_limits_like_llm() -> None:
    select, llm = service_block("llm-select"), service_block("llm")
    for key in ("cpus", "mem_limit", "memswap_limit"):
        assert _lines(select, key) == _lines(llm, key) != []
    assert "restart" not in select
    assert "      - ./select:/select:ro" in select


def test_selector_profile_pid_and_volume() -> None:
    block = service_block("selector")
    assert '    profiles: ["select"]' in block
    assert '    pid: "service:llm-select"' in block
    assert "      - models:/models:ro" in block
    assert re.search(r"depends_on:\n      llm-select:\n        condition: service_healthy", block)
    assert "restart" not in block
    assert "ports" not in block


def test_llm_and_api_have_no_profile() -> None:
    for name in ("llm", "api"):
        assert "profiles" not in service_block(name)
