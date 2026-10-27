"""SSE / streaming helpers. OWNER: proxy."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass


@dataclass(slots=True)
class SseEvent:
    event: str | None
    data: str
    raw: bytes  # exact bytes incl. trailing blank line, for lossless re-emission


_EOL = rb"(?:\r\n|\r|\n)"
_TERMINATOR = re.compile(_EOL + _EOL)


def _parse_event(raw: bytes) -> SseEvent:
    event: str | None = None
    data: list[str] = []
    for line in re.split(r"\r\n|\r|\n", raw.decode("utf-8", errors="replace")):
        if not line or line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    return SseEvent(event, "\n".join(data), raw)


class SseParser:
    """Incremental SSE framing: feed bytes, get complete events (handles splits and CRLF)."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> list[SseEvent]:
        self._buf += chunk
        out: list[SseEvent] = []
        while True:
            m = _TERMINATOR.search(self._buf)
            if m is None:
                break
            # A trailing CR may be the first half of CRLF: wait for more bytes to decide.
            if m.end() == len(self._buf) and self._buf.endswith(b"\r"):
                break
            out.append(_parse_event(self._buf[: m.end()]))
            self._buf = self._buf[m.end() :]
        return out

    def finish(self) -> list[SseEvent]:
        rest, self._buf = self._buf, b""
        return [_parse_event(rest)] if rest.strip(b"\r\n") else []


async def iter_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[SseEvent]:
    """Incrementally parse an SSE byte stream (handles events split across chunks, CRLF)."""
    parser = SseParser()
    try:
        async for chunk in chunks:
            for ev in parser.feed(chunk):
                yield ev
        for ev in parser.finish():
            yield ev
    finally:
        await aclose_quietly(chunks)


async def aclose_quietly(it: object) -> None:
    close = getattr(it, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:
            return


async def passthrough_with_tap(
    chunks: AsyncIterator[bytes], *, on_event: Callable[[SseEvent], None] | None = None
) -> AsyncIterator[bytes]:
    """Re-emit chunks unchanged and immediately (no added latency) while feeding parsed events
    to `on_event`. Propagates cancellation/aclose to the source."""
    parser = SseParser()
    try:
        async for chunk in chunks:
            yield chunk  # emit first: parsing never delays the client
            if on_event is not None:
                for ev in parser.feed(chunk):
                    on_event(ev)
        if on_event is not None:
            for ev in parser.finish():
                on_event(ev)
    finally:
        await aclose_quietly(chunks)
