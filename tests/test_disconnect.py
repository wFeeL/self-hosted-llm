import asyncio

import pytest
from starlette.types import Message

from error_classifier.api.disconnect import ClientDisconnectedError, cancel_on_disconnect
from error_classifier.errors import InvalidInputError

pytestmark = pytest.mark.anyio

DISCONNECT: Message = {"type": "http.disconnect"}
REQUEST: Message = {"type": "http.request", "body": b"", "more_body": False}


class FakeReceive:
    """Hands out the given messages one by one, each after `gate` is set (it closes again)."""

    def __init__(self, *messages: Message) -> None:
        self._messages = list(messages)
        self.gate = asyncio.Event()
        self.cancelled = False

    async def __call__(self) -> Message:
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.gate.clear()
        return self._messages.pop(0)


class Work:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancelled = False

    async def run(self) -> str:
        self.started.set()
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "done"


async def test_result_when_work_finishes_first() -> None:
    receive = FakeReceive(DISCONNECT)
    work = Work()
    work.finish.set()
    assert await cancel_on_disconnect(receive, work.run()) == "done"
    assert receive.cancelled


async def test_work_error_propagates() -> None:
    async def fail() -> None:
        raise InvalidInputError("bad")

    receive = FakeReceive(DISCONNECT)
    with pytest.raises(InvalidInputError):
        await cancel_on_disconnect(receive, fail())
    assert receive.cancelled


async def test_disconnect_cancels_work() -> None:
    receive = FakeReceive(DISCONNECT)
    work = Work()
    task = asyncio.create_task(cancel_on_disconnect(receive, work.run()))
    await work.started.wait()
    receive.gate.set()
    with pytest.raises(ClientDisconnectedError):
        await task
    assert work.cancelled


async def test_other_messages_are_ignored() -> None:
    receive = FakeReceive(REQUEST, DISCONNECT)
    work = Work()
    task = asyncio.create_task(cancel_on_disconnect(receive, work.run()))
    await work.started.wait()
    receive.gate.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not work.cancelled
    assert not task.done()
    receive.gate.set()
    with pytest.raises(ClientDisconnectedError):
        await task
    assert work.cancelled


async def test_caller_cancellation_cancels_both() -> None:
    receive = FakeReceive(DISCONNECT)
    work = Work()
    before = asyncio.all_tasks()
    task = asyncio.create_task(cancel_on_disconnect(receive, work.run()))
    await work.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert work.cancelled
    assert receive.cancelled
    assert [t for t in asyncio.all_tasks() - before if not t.done()] == []
