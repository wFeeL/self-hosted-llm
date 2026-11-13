from typing import cast

import httpx

from error_classifier.config import Settings
from error_classifier.errors import LlmUnavailableError


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    # No default timeout: every call sets its own (probe here, generation in F1).
    return httpx.AsyncClient(base_url=settings.llm_base_url + "/", timeout=None)


class LlmClient:
    def __init__(self, http: httpx.AsyncClient, model: str, probe_timeout_s: float) -> None:
        self._http = http
        self.model = model
        self.probe_timeout_s = probe_timeout_s

    async def check_ready(self) -> None:
        try:
            response = await self._http.get("models", timeout=self.probe_timeout_s)
        except httpx.TimeoutException as exc:
            raise LlmUnavailableError(
                f"LLM runtime did not answer within {self.probe_timeout_s} s"
            ) from exc
        except httpx.TransportError as exc:
            raise LlmUnavailableError("LLM runtime is unreachable") from exc
        if response.status_code == 503:
            raise LlmUnavailableError("LLM runtime is still loading the model")
        if response.status_code != 200:
            raise LlmUnavailableError(f"LLM runtime returned HTTP {response.status_code}")
        if self.model not in _model_ids(response):
            raise LlmUnavailableError(f"Model '{self.model}' is not served by the LLM runtime")


def _model_ids(response: httpx.Response) -> list[str]:
    unexpected = LlmUnavailableError("LLM runtime returned an unexpected model list")
    try:
        body = cast(object, response.json())
    except ValueError as exc:
        raise unexpected from exc
    if not isinstance(body, dict):
        raise unexpected
    data = cast(dict[str, object], body).get("data")
    if not isinstance(data, list):
        raise unexpected
    ids: list[str] = []
    for item in cast(list[object], data):
        model_id = cast(dict[str, object], item).get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str):
            raise unexpected
        ids.append(model_id)
    return ids
