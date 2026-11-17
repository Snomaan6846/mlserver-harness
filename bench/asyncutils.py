"""Async/sync bridge utilities.

The suites are async coroutines (started once per suite via
``asyncio.run``), but some components — the container lifecycle
(``MLServerContainer._wait_healthy``) and the per-case load runs
(``run_closed``) — are synchronous APIs that need to drive coroutines
to completion. Calling ``asyncio.run`` from inside a running loop is
forbidden, so these helpers detect the context and fall back to a
dedicated worker thread with its own event loop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine


def run_coroutine_blocking(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine to completion from synchronous code.

    - No running loop in this thread: plain ``asyncio.run``.
    - Inside a running loop (the suite coroutines): execute the
      coroutine in a single-use worker thread with its own loop and
      block until it finishes.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # We are inside a running event loop: a dedicated thread gets its
    # own loop, so the coroutine (and any loop-bound resources it
    # creates — aiohttp sessions, gRPC channels) run cleanly there.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()
