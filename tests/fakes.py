import asyncio
import json
from collections.abc import Mapping, Sequence

from error_classifier.llm.client import ChatCompletion, ChatMessage

ANSWER: dict[str, str] = {
    "category": "network_error",
    "severity": "high",
    "summary": "The upstream refused the connection.",
    "suggested_action": "Check that the upstream is running.",
}


class BlockingChat:
    """ChatClient fake: every call waits for `release`, the first `errors` raise at once."""

    def __init__(self, errors: Sequence[Exception] = ()) -> None:
        self._errors = list(errors)
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.calls = 0
        self.cancelled = 0

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, object],
        schema_name: str,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        disable_thinking: bool,
    ) -> ChatCompletion:
        self.calls += 1
        self.started.set()
        if self._errors:
            raise self._errors.pop(0)
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return ChatCompletion(json.dumps(ANSWER), "stop", "granite", 11, 7)
