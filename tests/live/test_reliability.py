import asyncio
import base64
import json
import logging
import os
import random
import sys
import time
from collections.abc import Awaitable
from pathlib import Path

import anyio
import httpx
import pytest
import uvicorn
from conftest import Serve
from pydantic import TypeAdapter

from error_classifier.api.app import create_app
from error_classifier.config import Settings
from error_classifier.evaluation.metrics import DEFAULT_MODEL
from error_classifier.evaluation.results import read_result

# Read at import, before the autouse clean_env fixture runs.
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", DEFAULT_MODEL)
# The default follows the evaluation run of the default model, so a model that thinks gets it off.
_EVALUATED = read_result(
    Path(__file__).resolve().parents[2] / "eval/results" / f"{DEFAULT_MODEL}.json"
).meta.settings.disable_thinking
DISABLE_THINKING = TypeAdapter(bool).validate_python(
    os.environ.get("LLM_DISABLE_THINKING", _EVALUATED)
)

pytestmark = [pytest.mark.live, pytest.mark.anyio]

LLM_ROOT = LLM_URL.rstrip("/").removesuffix("/v1")
SHORT_MESSAGE = "TypeError: Cannot read properties of undefined (reading 'map')"

# On macOS a cancelled httpx request leaves its socket open, so a service run on the host never
# tells llama-server to stop; the same service inside the Linux container of the stack does.
_HOST_CANNOT_CANCEL = pytest.mark.xfail(
    sys.platform == "darwin",
    reason="httpx on macOS does not close a cancelled connection (Python 3.14.10, 2026-11-21)",
    run=False,
)


def _live_settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "llm_base_url": LLM_URL,
        "llm_model": MODEL_ALIAS,
        "llm_disable_thinking": DISABLE_THINKING,
    }
    return Settings.model_validate({**values, **changes})


# Each test gets its own seed so the slot's prompt cache cannot hand the next request a prefix.
def _base64_message(seed: int) -> str:
    return base64.b64encode(random.Random(seed).randbytes(2400)).decode()[:3000]


async def _wait_for_free_slots(limit_s: float, what: str) -> float:
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=5) as http:
        while True:
            slots = (await http.get(f"{LLM_ROOT}/slots")).json()
            if not any(slot["is_processing"] for slot in slots):
                return time.perf_counter() - started
            if time.perf_counter() - started > limit_s:
                pytest.fail(f"llama-server slots still busy {limit_s:.1f} s {what}")
            await asyncio.sleep(0.25)


async def _run_until_busy[T](work: Awaitable[T]) -> tuple[T, bool]:
    # A 504 or a disconnect proves nothing about cancellation if the prompt never reached a slot.
    task = asyncio.ensure_future(work)
    busy = False
    async with httpx.AsyncClient(timeout=5) as http:
        while not task.done():
            slots = (await http.get(f"{LLM_ROOT}/slots")).json()
            busy = busy or any(slot["is_processing"] for slot in slots)
            await asyncio.wait({task}, timeout=0.25)
    return await task, busy


def _require_long_reference(reference: float, before: float) -> None:
    if reference < 2 * before:
        pytest.fail(
            f"the reference run took {reference:.1f} s, "
            "too short to tell a cancelled prompt from a finished one"
        )


_REFERENCE_S: dict[str, float] = {}


async def _reference_s(serve: Serve) -> float:
    if "base64" in _REFERENCE_S:
        return _REFERENCE_S["base64"]
    await _wait_for_free_slots(600, "before the reference run")
    app = create_app(_live_settings(request_timeout_s=600))
    async with serve(app) as client:
        started = time.perf_counter()
        response = await _classify(client, "reference", _base64_message(10))
        elapsed = time.perf_counter() - started
    if response.status_code != 200:
        pytest.fail(f"reference run answered {response.status_code}: {response.text}")
    _REFERENCE_S["base64"] = elapsed
    return elapsed


async def _classify(client: httpx.AsyncClient, label: str, message: str) -> httpx.Response:
    started = time.perf_counter()
    response = await client.post("/classify", json={"message": message})
    elapsed = round(time.perf_counter() - started, 1)
    # One line per request for the run log; read it with pytest -s.
    print(
        json.dumps(
            {
                "case": label,
                "status": response.status_code,
                "seconds": elapsed,
                "body": response.text,
            }
        )
    )
    return response


async def test_second_request_gets_429(serve: Serve) -> None:
    app = create_app(_live_settings(max_concurrency=1, max_queue=0))
    async with serve(app) as client:
        first = asyncio.create_task(_classify(client, "first", SHORT_MESSAGE))
        try:
            with anyio.fail_after(5):
                while app.state.limiter.active != 1:  # noqa: ASYNC110
                    await asyncio.sleep(0.05)
            second = await _classify(client, "second", SHORT_MESSAGE)
        finally:
            response = await first
    assert second.status_code == 429
    assert "retry-after" in second.headers
    assert response.status_code == 200
    body = response.json()
    for field in ("category", "severity", "summary", "suggested_action"):
        assert body[field]


@_HOST_CANNOT_CANCEL
async def test_deadline_gives_504_and_frees_the_slot(serve: Serve) -> None:
    reference = await _reference_s(serve)
    _require_long_reference(reference, 3)
    await _wait_for_free_slots(reference, "after the previous test")
    app = create_app(_live_settings(request_timeout_s=3))
    async with serve(app) as client:
        started = time.perf_counter()
        response, busy = await _run_until_busy(_classify(client, "deadline", _base64_message(11)))
        elapsed = time.perf_counter() - started
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "llm_timeout"
        assert elapsed <= 4.5
        if not busy:
            pytest.fail("the request never reached llama-server: no slot was busy before the 504")
        budget = (reference - elapsed) / 2
        released = await _wait_for_free_slots(reference, "after the 504")
    assert released <= budget, (
        f"slot busy {released:.1f} s after the 504, budget {budget:.1f} s, "
        f"a full run of the same input takes {reference:.1f} s"
    )


@_HOST_CANNOT_CANCEL
async def test_disconnect_frees_the_slot(serve: Serve, caplog: pytest.LogCaptureFixture) -> None:
    reference = await _reference_s(serve)
    _require_long_reference(reference, 3)
    await _wait_for_free_slots(reference, "after the previous test")
    caplog.set_level(logging.INFO, logger="error_classifier")

    def closed() -> bool:
        return any("outcome=client_closed" in r.getMessage() for r in caplog.records)

    app = create_app(_live_settings(request_timeout_s=120))
    # A short graceful shutdown makes a failing run cancel the work instead of waiting for it.
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=5
    )
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    try:
        with anyio.fail_after(10):
            while not server.started:  # noqa: ASYNC110
                await asyncio.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=None) as client:
            request = asyncio.create_task(
                client.post("/classify", json={"message": _base64_message(12)})
            )
            _, busy = await _run_until_busy(asyncio.sleep(3))
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        left = time.perf_counter()
        if not busy:
            pytest.fail(
                "the request never reached llama-server: no slot was busy before the client left"
            )
        with anyio.move_on_after(10):
            while not closed():  # noqa: ASYNC110
                await asyncio.sleep(0.1)
        if not closed():
            pytest.fail(
                "the API did not log outcome=client_closed within 10 s after the client left"
            )
        await _wait_for_free_slots(reference, "after the client left")
        released = time.perf_counter() - left
        budget = (reference - 3) / 2
        assert released <= budget, (
            f"slot busy {released:.1f} s after the client left, budget {budget:.1f} s, "
            f"a full run of the same input takes {reference:.1f} s"
        )
    finally:
        server.should_exit = True
        await serving
