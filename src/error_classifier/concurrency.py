import asyncio
import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Self

from error_classifier.config import Settings
from error_classifier.errors import QueueFullError


class ConcurrencyLimiter:
    def __init__(self, max_concurrency: int, max_queue: int, queue_timeout_s: float) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_queue = max_queue
        self._queue_timeout_s = queue_timeout_s
        self.active = 0  # requests holding a slot
        self.waiting = 0  # requests waiting for a slot
        self.retry_after_s = max(1, math.ceil(queue_timeout_s))

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.max_concurrency, settings.max_queue, settings.queue_timeout_s)

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None]:
        if self._semaphore.locked():
            # locked() is also true while others wait, so a newcomer never jumps the queue.
            if self.waiting >= self._max_queue:
                raise QueueFullError(
                    "All model slots are busy and the queue is full, retry later",
                    self.retry_after_s,
                )
            self.waiting += 1
            try:
                async with asyncio.timeout(self._queue_timeout_s):
                    await self._semaphore.acquire()
            except TimeoutError:
                raise QueueFullError(
                    f"No model slot became free within {self._queue_timeout_s:g} s, retry later",
                    self.retry_after_s,
                ) from None
            finally:
                self.waiting -= 1
        else:
            await self._semaphore.acquire()
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1
            self._semaphore.release()
