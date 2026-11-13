import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_PROBE_TIMEOUT_S",
        "LOG_LEVEL",
        "API_URL",
        "MODEL_ALIAS",
    ):
        monkeypatch.delenv(name, raising=False)
