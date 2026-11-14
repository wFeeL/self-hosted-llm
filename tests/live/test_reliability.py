import asyncio
import json
import os
import time

import anyio
import httpx
import pytest
from conftest import Serve

from error_classifier.api.app import create_app
from error_classifier.config import Settings

# Read at import, before the autouse clean_env fixture runs.
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "granite-4.1-3b-q4km")

pytestmark = [pytest.mark.live, pytest.mark.anyio]

SHORT_MESSAGE = "TypeError: Cannot read properties of undefined (reading 'map')"


def _live_settings(**changes: object) -> Settings:
    values: dict[str, object] = {"llm_base_url": LLM_URL, "llm_model": MODEL_ALIAS}
    return Settings.model_validate({**values, **changes})


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
