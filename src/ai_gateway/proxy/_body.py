"""Response body wrappers. OWNER: proxy."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace

from ai_gateway.errors import UpstreamError
from ai_gateway.proxy.base import ProxyResponse
from ai_gateway.proxy.streaming import aclose_quietly

log = logging.getLogger(__name__)

# Upper bound for bodies the gateway must hold in memory to inspect/rewrite them.
MAX_BUFFERED_BYTES = 8 * 1024 * 1024


class TapBody:
    """Async iterator that relays chunks unchanged while observing them.

    `on_done` is awaited exactly once: when the source is exhausted, raises, or `aclose()` is
    called (client disconnect), even if iteration never started.
    """

    def __init__(
        self,
        source: AsyncIterator[bytes],
        *,
        on_chunk: Callable[[bytes], None] | None,
        on_done: Callable[[], Awaitable[None]],
    ) -> None:
        self._source = source
        self._on_chunk = on_chunk
        self._on_done = on_done
        self._done = False

    def __aiter__(self) -> TapBody:
        return self

    async def __anext__(self) -> bytes:
        if self._done:
            raise StopAsyncIteration
        try:
            chunk = await self._source.__anext__()
        except BaseException:
            await self._finish()
            raise
        if self._on_chunk is not None:
            try:
                self._on_chunk(chunk)
            except Exception:
                log.debug("tap failed", exc_info=True)
                self._on_chunk = None
        return chunk

    async def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            await self._on_done()
        except Exception:
            log.warning("response finalizer failed", exc_info=True, extra={"event": "proxy.finalizer"})

    async def aclose(self) -> None:
        await self._finish()
        await aclose_quietly(self._source)


def with_body(resp: ProxyResponse, body: AsyncIterator[bytes]) -> ProxyResponse:
    """Copy of `resp` with a new body whose aclose() also closes the original upstream stream."""
    orig_close = resp._close

    async def close() -> None:
        await aclose_quietly(body)
        if orig_close is not None:
            await orig_close()

    return replace(resp, body=body, _close=close)


async def read_bounded(chunks: AsyncIterator[bytes], limit: int = MAX_BUFFERED_BYTES) -> bytes:
    buf = bytearray()
    async for chunk in chunks:
        buf += chunk
        if len(buf) > limit:
            raise UpstreamError("Upstream response too large to inspect", code="upstream_error")
    return bytes(buf)
