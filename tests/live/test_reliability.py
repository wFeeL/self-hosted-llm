import asyncio
import base64
import json
import logging
import os
import random
import time

import anyio
import httpx
import pytest
import uvicorn
from conftest import Serve
from pydantic import TypeAdapter

from error_classifier.api.app import create_app
from error_classifier.config import Settings
from error_classifier.evaluation.metrics import DEFAULT_MODEL

# Read at import, before the autouse clean_env fixture runs.
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", DEFAULT_MODEL)
DISABLE_THINKING = TypeAdapter(bool).validate_python(os.environ.get("LLM_DISABLE_THINKING", "true"))

pytestmark = [pytest.mark.live, pytest.mark.anyio]

LLM_ROOT = LLM_URL.rstrip("/").removesuffix("/v1")
SHORT_MESSAGE = "TypeError: Cannot read properties of undefined (reading 'map')"


def _live_settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "llm_base_url": LLM_URL,
        "llm_model": MODEL_ALIAS,
        "llm_disable_thinking": DISABLE_THINKING,
    }
    return Settings.model_validate({**values, **changes})


def _base64_message() -> str:
    return base64.b64encode(random.Random(7).randbytes(2400)).decode()[:3000]


async def _wait_for_free_slots(limit_s: float) -> None:
    with anyio.fail_after(limit_s):
        async with httpx.AsyncClient(timeout=5) as http:
            while True:
                slots = (await http.get(f"{LLM_ROOT}/slots")).json()
                if not any(slot["is_processing"] for slot in slots):
                    return
                await asyncio.sleep(0.25)


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


async def test_deadline_gives_504_and_frees_the_slot(serve: Serve) -> None:
    app = create_app(_live_settings(request_timeout_s=3))
    async with serve(app) as client:
        started = time.perf_counter()
        response = await _classify(client, "deadline", _base64_message())
        elapsed = time.perf_counter() - started
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "llm_timeout"
        assert elapsed <= 4.5
        await _wait_for_free_slots(15)


async def test_disconnect_frees_the_slot(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="error_classifier")
    app = create_app(_live_settings(request_timeout_s=120))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    try:
        with anyio.fail_after(10):
            while not server.started:  # noqa: ASYNC110
                await asyncio.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=None) as client:
            request = asyncio.create_task(
                client.post("/classify", json={"message": _base64_message()})
            )
            await asyncio.sleep(3)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        await _wait_for_free_slots(15)
    finally:
        server.should_exit = True
        await serving
    assert any("outcome=client_closed" in r.getMessage() for r in caplog.records)
