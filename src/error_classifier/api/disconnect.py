import asyncio
from collections.abc import Coroutine

from starlette.types import Receive


class ClientDisconnectedError(Exception): ...


async def _wait_for_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


async def cancel_on_disconnect[T](receive: Receive, work: Coroutine[object, object, T]) -> T:
    work_task = asyncio.create_task(work)
    watcher = asyncio.create_task(_wait_for_disconnect(receive))
    try:
        await asyncio.wait({work_task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # The caller was cancelled: no task may outlive the request and keep the slot.
        work_task.cancel()
        watcher.cancel()
        await asyncio.wait({work_task, watcher})
        raise
    if work_task.done():
        watcher.cancel()
        # asyncio.wait does not swallow a cancellation of the caller, unlike suppress().
        await asyncio.wait({watcher})
        return work_task.result()
    work_task.cancel()
    await asyncio.wait({work_task})
    if not work_task.cancelled():
        work_task.exception()  # mark retrieved: the client is gone either way
    raise ClientDisconnectedError
