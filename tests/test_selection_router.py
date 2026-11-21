import json
from typing import Any

import httpx
import pytest
import respx

from error_classifier.selection.router import (
    LoadOutcome,
    ModelStatus,
    RouterClient,
    RouterError,
    main_llm_running,
)

pytestmark = pytest.mark.anyio

BASE = "http://router:8080"


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _entry(
    alias: str,
    value: str,
    *,
    failed: object = None,
    exit_code: object = None,
    source: str = "preset",
) -> dict[str, Any]:
    status: dict[str, Any] = {"value": value}
    if failed is not None:
        status["failed"] = failed
    if exit_code is not None:
        status["exit_code"] = exit_code
    return {"id": alias, "source": source, "status": status}


def _models(*entries: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"data": list(entries)})


def _client(http: httpx.AsyncClient, time: FakeTime, **kwargs: float) -> RouterClient:
    return RouterClient(http, sleep=time.sleep, clock=time.clock, **kwargs)


async def test_statuses_keeps_presets_only() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.get("/models").mock(
            return_value=_models(
                _entry("a", "unloaded", failed=True, exit_code=1),
                _entry("b", "loaded", exit_code=True),
                _entry("c", "loaded", source="cache"),
            )
        )
        async with httpx.AsyncClient(base_url=BASE) as http:
            got = await _client(http, FakeTime()).statuses()
    assert got == {
        "a": ModelStatus("unloaded", True, 1),
        "b": ModelStatus("loaded", False, None),
    }


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"models": []}),
        httpx.Response(200, json={"data": [{"id": "a", "source": "preset", "status": {}}]}),
        httpx.ConnectError("down"),
    ],
)
async def test_statuses_unexpected_shape_raises(response: httpx.Response | Exception) -> None:
    with respx.mock(base_url=BASE) as mock:
        route = mock.get("/models")
        if isinstance(response, Exception):
            route.mock(side_effect=response)
        else:
            route.mock(return_value=response)
        async with httpx.AsyncClient(base_url=BASE) as http:
            with pytest.raises(RouterError):
                await _client(http, FakeTime()).statuses()


async def test_load_waits_until_loaded() -> None:
    time = FakeTime()
    with respx.mock(base_url=BASE) as mock:
        post = mock.post("/models/load").mock(return_value=httpx.Response(200, json={}))
        get = mock.get("/models").mock(
            side_effect=[
                _models(_entry("a", "downloading")),
                _models(_entry("a", "loading")),
                _models(_entry("a", "loaded")),
            ]
        )
        async with httpx.AsyncClient(base_url=BASE) as http:
            outcome = await _client(http, time, poll_s=2.0).load("a", 600)
    assert outcome == LoadOutcome(True, 4.0, None, None)
    assert get.call_count == 3
    assert json.loads(post.calls.last.request.content) == {"model": "a"}


async def test_load_reports_failure_with_exit_code() -> None:
    time = FakeTime()
    with respx.mock(base_url=BASE) as mock:
        mock.post("/models/load").mock(return_value=httpx.Response(200, json={}))
        mock.get("/models").mock(
            return_value=_models(_entry("a", "unloaded", failed=True, exit_code=1))
        )
        async with httpx.AsyncClient(base_url=BASE) as http:
            outcome = await _client(http, time).load("a", 600)
    assert not outcome.ok
    assert outcome.exit_code == 1
    assert outcome.reason == "failed to load (exit code 1)"
    assert "memory" not in (outcome.reason or "")


async def test_load_failure_without_exit_code() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/models/load").mock(return_value=httpx.Response(200, json={}))
        mock.get("/models").mock(return_value=_models(_entry("a", "unloaded", failed=True)))
        async with httpx.AsyncClient(base_url=BASE) as http:
            outcome = await _client(http, FakeTime()).load("a", 600)
    assert outcome.reason == "failed to load"
    assert outcome.exit_code is None


async def test_load_times_out() -> None:
    time = FakeTime()
    with respx.mock(base_url=BASE) as mock:
        mock.post("/models/load").mock(return_value=httpx.Response(200, json={}))
        mock.get("/models").mock(return_value=_models(_entry("a", "loading")))
        async with httpx.AsyncClient(base_url=BASE) as http:
            outcome = await _client(http, time, poll_s=2.0).load("a", 10)
    assert not outcome.ok
    assert outcome.reason == "not downloaded or loaded within 10 s"
    assert outcome.seconds == 10.0


async def test_load_alias_not_listed() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/models/load").mock(return_value=httpx.Response(200, json={}))
        mock.get("/models").mock(return_value=_models(_entry("other", "loaded")))
        async with httpx.AsyncClient(base_url=BASE) as http:
            outcome = await _client(http, FakeTime()).load("a", 600)
    assert outcome.reason == "not listed by the router"


async def test_load_refused() -> None:
    with respx.mock(base_url=BASE) as mock:
        post = mock.post("/models/load").mock(
            return_value=httpx.Response(404, text="no such model")
        )
        async with httpx.AsyncClient(base_url=BASE) as http:
            outcome = await _client(http, FakeTime()).load("a", 600)
        total_calls = mock.calls.call_count
    assert outcome == LoadOutcome(False, 0.0, "router refused to load: HTTP 404", None)
    assert total_calls == 1
    assert post.called


async def test_load_already_loaded_continues() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/models/load").mock(
            return_value=httpx.Response(
                400, json={"error": {"message": "Model is Already running"}}
            )
        )
        mock.get("/models").mock(return_value=_models(_entry("a", "loaded")))
        async with httpx.AsyncClient(base_url=BASE) as http:
            outcome = await _client(http, FakeTime()).load("a", 600)
    assert outcome.ok


async def test_load_propagates_router_error_while_polling() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/models/load").mock(return_value=httpx.Response(200, json={}))
        mock.get("/models").mock(return_value=httpx.Response(500))
        async with httpx.AsyncClient(base_url=BASE) as http:
            with pytest.raises(RouterError):
                await _client(http, FakeTime()).load("a", 600)


async def test_unload_waits() -> None:
    time = FakeTime()
    with respx.mock(base_url=BASE) as mock:
        post = mock.post("/models/unload").mock(return_value=httpx.Response(200, json={}))
        mock.get("/models").mock(
            side_effect=[_models(_entry("a", "loaded")), _models(_entry("a", "unloaded"))]
        )
        async with httpx.AsyncClient(base_url=BASE) as http:
            assert await _client(http, time, poll_s=2.0).unload("a") is True
    assert json.loads(post.calls.last.request.content) == {"model": "a"}


async def test_unload_times_out() -> None:
    time = FakeTime()
    with respx.mock(base_url=BASE) as mock:
        mock.post("/models/unload").mock(return_value=httpx.Response(200, json={}))
        mock.get("/models").mock(return_value=_models(_entry("a", "loaded")))
        async with httpx.AsyncClient(base_url=BASE) as http:
            assert await _client(http, time, poll_s=2.0).unload("a", timeout_s=10) is False
    assert time.now == 110.0


async def test_unload_alias_missing_is_false() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/models/unload").mock(return_value=httpx.Response(200, json={}))
        mock.get("/models").mock(return_value=_models())
        async with httpx.AsyncClient(base_url=BASE) as http:
            assert await _client(http, FakeTime()).unload("a") is False


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (httpx.Response(200), True),
        (httpx.Response(503), True),
        (httpx.ConnectError("refused"), False),
        (httpx.ReadTimeout("slow"), False),
    ],
)
async def test_main_llm_running(result: httpx.Response | Exception, expected: bool) -> None:
    with respx.mock() as mock:
        route = mock.get("http://llm:8080/health")
        if isinstance(result, Exception):
            route.mock(side_effect=result)
        else:
            route.mock(return_value=result)
        async with httpx.AsyncClient() as http:
            assert await main_llm_running(http, "http://llm:8080") is expected
