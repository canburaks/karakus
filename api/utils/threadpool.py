from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


async def run_in_threadpool(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a sync function in a threadpool"""
    loop = get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, lambda: func(*args, **kwargs))
