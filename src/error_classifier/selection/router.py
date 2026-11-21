import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import httpx


@dataclass(frozen=True)
class ModelStatus:
    value: str  # status.value as reported
    failed: bool  # status.failed is True
    exit_code: int | None  # status.exit_code when int


@dataclass(frozen=True)
class LoadOutcome:
    ok: bool
    seconds: float
    reason: str | None  # None when ok
    exit_code: int | None


class RouterError(Exception):
    """The router is unreachable or answered in an unexpected shape."""


def _parse_entry(entry: object) -> tuple[str, ModelStatus] | None:
    if not isinstance(entry, dict):
        raise RouterError("the router listed a model that is not an object")
    fields = cast("dict[str, object]", entry)
    if fields.get("source") != "preset":
        return None
    alias = fields.get("id")
    status = fields.get("status")
    if not isinstance(alias, str) or not isinstance(status, dict):
        raise RouterError("the router listed a preset without an id or a status")
    status_fields = cast("dict[str, object]", status)
    value = status_fields.get("value")
    if not isinstance(value, str):
        raise RouterError(f"the router gave no status value for {alias}")
    exit_code = status_fields.get("exit_code")
    return alias, ModelStatus(
        value=value,
        failed=status_fields.get("failed") is True,
        exit_code=exit_code
        if isinstance(exit_code, int) and not isinstance(exit_code, bool)
        else None,
    )


class RouterClient:
    def __init__(
        self,
        http: httpx.AsyncClient,  # base_url is the router root
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = 2.0,
        call_timeout_s: float = 10.0,
    ) -> None:
        self._http = http
        self._sleep = sleep
        self._clock = clock
        self._poll_s = poll_s
        self._call_timeout_s = call_timeout_s

    async def statuses(self) -> dict[str, ModelStatus]:
        try:
            response = await self._http.get("/models", timeout=self._call_timeout_s)
        except httpx.HTTPError as exc:
            raise RouterError(f"the router is unreachable: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise RouterError(f"the router answered HTTP {response.status_code} to the model list")
        try:
            body: object = response.json()
        except ValueError as exc:
            raise RouterError("the router answered the model list with invalid JSON") from exc
        data = cast("dict[str, object]", body).get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise RouterError("the router answered the model list without a data array")
        result: dict[str, ModelStatus] = {}
        for entry in cast("list[object]", data):
            parsed = _parse_entry(entry)
            if parsed is not None:
                result[parsed[0]] = parsed[1]
        return result

    async def _post(self, path: str, alias: str) -> httpx.Response:
        try:
            return await self._http.post(path, json={"model": alias}, timeout=self._call_timeout_s)
        except httpx.HTTPError as exc:
            raise RouterError(f"the router is unreachable: {type(exc).__name__}") from exc

    async def load(self, alias: str, timeout_s: float) -> LoadOutcome:
        t0 = self._clock()
        deadline = t0 + timeout_s
        response = await self._post("/models/load", alias)
        # llama-server answers 400 "model is already running" for a loaded model.
        already = response.status_code == 400 and "already" in response.text.lower()
        if response.status_code != 200 and not already:
            return LoadOutcome(
                False,
                self._clock() - t0,
                f"router refused to load: HTTP {response.status_code}",
                None,
            )
        while True:
            status = (await self.statuses()).get(alias)
            if status is None:
                return LoadOutcome(False, self._clock() - t0, "not listed by the router", None)
            if status.value == "loaded":
                return LoadOutcome(True, self._clock() - t0, None, None)
            if status.failed:
                reason = "failed to load"
                if status.exit_code is not None:
                    reason += f" (exit code {status.exit_code})"
                return LoadOutcome(False, self._clock() - t0, reason, status.exit_code)
            if self._clock() >= deadline:
                reason = f"not downloaded or loaded within {timeout_s:g} s"
                return LoadOutcome(False, self._clock() - t0, reason, None)
            await self._sleep(self._poll_s)

    async def unload(self, alias: str, timeout_s: float = 60.0) -> bool:
        deadline = self._clock() + timeout_s
        await self._post("/models/unload", alias)
        while True:
            status = (await self.statuses()).get(alias)
            if status is None:
                return False
            if status.value == "unloaded":
                return True
            if self._clock() >= deadline:
                return False
            await self._sleep(self._poll_s)


async def main_llm_running(http: httpx.AsyncClient, url: str, timeout_s: float = 2.0) -> bool:
    try:
        await http.get(f"{url}/health", timeout=timeout_s)
    except httpx.TransportError:
        return False
    return True
