from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
import pytest
import respx
from bench_fakes import Stack
from fastapi import FastAPI

Serve = Callable[[FastAPI], AbstractAsyncContextManager[httpx.AsyncClient]]


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
        "REQUEST_TIMEOUT_S",
        "MAX_CONCURRENCY",
        "MAX_QUEUE",
        "QUEUE_TIMEOUT_S",
        "LLM_URL",
        "MAX_INPUT_CHARS",
        "MAX_REQUEST_BYTES",
        "MAX_OUTPUT_TOKENS",
        "LLM_TEMPERATURE",
        "LLM_DISABLE_THINKING",
        "API_URL",
        "MODEL_ALIAS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def serve() -> Serve:
    # starlette's TestClient is typed against httpx2, so tests talk to the app over httpx.
    @asynccontextmanager
    async def _serve(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient]:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://api.test"
            ) as client,
        ):
            yield client

    return _serve


@pytest.fixture
def stack() -> Iterator[Stack]:
    with respx.mock(assert_all_called=False) as router:
        yield Stack(router)
