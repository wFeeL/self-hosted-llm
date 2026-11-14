import asyncio
import time

import anyio
import pytest

from error_classifier.concurrency import ConcurrencyLimiter
from error_classifier.config import Settings
from error_classifier.errors import QueueFullError

pytestmark = pytest.mark.anyio


def _limiter(
    max_concurrency: int = 1, max_queue: int = 1, queue_timeout_s: float = 5.0
) -> ConcurrencyLimiter:
    return ConcurrencyLimiter(max_concurrency, max_queue, queue_timeout_s)


async def _wait_for_waiting(limiter: ConcurrencyLimiter, count: int) -> None:
    with anyio.fail_after(1):
        while limiter.waiting != count:  # noqa: ASYNC110
            await asyncio.sleep(0)


async def _hold(limiter: ConcurrencyLimiter, entered: asyncio.Event, leave: asyncio.Event) -> None:
    async with limiter.slot():
        entered.set()
        await leave.wait()


async def test_free_slot_enters_at_once() -> None:
    limiter = _limiter()
    with anyio.fail_after(0.5):
        async with limiter.slot():
            assert limiter.active == 1
    assert limiter.active == 0


async def test_second_request_waits_for_the_first() -> None:
    limiter = _limiter()
    events: list[str] = []
    entered, leave = asyncio.Event(), asyncio.Event()

    async def second() -> None:
        async with limiter.slot():
            events.append("second in")

    holder = asyncio.create_task(_hold(limiter, entered, leave))
    await entered.wait()
    waiter = asyncio.create_task(second())
    await _wait_for_waiting(limiter, 1)
    events.append("first out")
    leave.set()
    await holder
    await waiter
    assert events == ["first out", "second in"]


async def test_full_queue_rejects_at_once() -> None:
    limiter = _limiter(queue_timeout_s=0.2)
    entered, leave = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(limiter, entered, leave))
    await entered.wait()

    async def queued() -> None:
        async with limiter.slot():
            pass

    waiter = asyncio.create_task(queued())
    await _wait_for_waiting(limiter, 1)
    started = time.perf_counter()
    with pytest.raises(QueueFullError) as info:
        async with limiter.slot():
            pass
    assert time.perf_counter() - started < 0.05
    assert info.value.retry_after_s == 1
    assert info.value.message == "All model slots are busy and the queue is full, retry later"
    assert limiter.waiting == 1
    leave.set()
    await holder
    await waiter


async def test_zero_queue_rejects_when_busy() -> None:
    limiter = _limiter(max_queue=0)
    entered, leave = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(limiter, entered, leave))
    await entered.wait()
    with pytest.raises(QueueFullError):
        async with limiter.slot():
            pass
    leave.set()
    await holder


async def test_wait_timeout_rejects() -> None:
    limiter = _limiter(queue_timeout_s=0.05)
    entered, leave = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(limiter, entered, leave))
    await entered.wait()
    with pytest.raises(QueueFullError) as info:
        async with limiter.slot():
            pass
    assert info.value.message == "No model slot became free within 0.05 s, retry later"
    assert limiter.waiting == 0
    leave.set()
    await holder


async def test_cancelled_waiter_leaves_the_queue() -> None:
    limiter = _limiter()
    entered, leave = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(limiter, entered, leave))
    await entered.wait()

    async def queued() -> None:
        async with limiter.slot():
            pass

    waiter = asyncio.create_task(queued())
    await _wait_for_waiting(limiter, 1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert limiter.waiting == 0
    leave.set()
    await holder
    with anyio.fail_after(0.5):
        async with limiter.slot():
            assert limiter.active == 1


async def test_error_inside_slot_releases_it() -> None:
    limiter = _limiter()
    with pytest.raises(RuntimeError, match="boom"):
        async with limiter.slot():
            raise RuntimeError("boom")
    assert limiter.active == 0
    with anyio.fail_after(0.5):
        async with limiter.slot():
            assert limiter.active == 1


async def test_waiters_enter_in_arrival_order() -> None:
    limiter = _limiter(max_queue=3)
    entered, leave = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(_hold(limiter, entered, leave))
    await entered.wait()
    order: list[int] = []

    async def queued(number: int) -> None:
        async with limiter.slot():
            order.append(number)

    waiters: list[asyncio.Task[None]] = []
    for number in range(3):
        waiters.append(asyncio.create_task(queued(number)))
        await _wait_for_waiting(limiter, number + 1)
    leave.set()
    await holder
    await asyncio.gather(*waiters)
    assert order == [0, 1, 2]


@pytest.mark.parametrize(("timeout_s", "expected"), [(30, 30), (30.2, 31), (0.05, 1)])
def test_retry_after_rounds_up(timeout_s: float, expected: int) -> None:
    assert _limiter(queue_timeout_s=timeout_s).retry_after_s == expected


def test_from_settings() -> None:
    settings = Settings(llm_model="m", max_concurrency=3, max_queue=0, queue_timeout_s=2.5)
    limiter = ConcurrencyLimiter.from_settings(settings)
    assert limiter.retry_after_s == 3
