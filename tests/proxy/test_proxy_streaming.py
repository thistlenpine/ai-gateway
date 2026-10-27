from __future__ import annotations

from collections.abc import AsyncIterator

from proxy_helpers import chunks

from ai_gateway.proxy.streaming import SseEvent, iter_sse, passthrough_with_tap


async def test_iter_sse_framing() -> None:
    raw = b"event: a\ndata: 1\ndata: 2\n\n: c\n\ndata: x\r\n\r\nid: 1\rdata: y\r\r"
    parts = [raw[i : i + 3] for i in range(0, len(raw), 3)]
    evs = [e async for e in iter_sse(chunks(*parts))]
    assert [(e.event, e.data) for e in evs] == [("a", "1\n2"), (None, ""), (None, "x"), (None, "y")]
    assert b"".join(e.raw for e in evs) == raw


async def test_iter_sse_flushes_unterminated_tail() -> None:
    evs = [e async for e in iter_sse(chunks(b"data: 1\n\ndata: tail"))]
    assert [e.data for e in evs] == ["1", "tail"]


async def test_passthrough_with_tap_unchanged_and_immediate() -> None:
    seen: list[SseEvent] = []
    out: list[bytes] = []
    closed = False

    async def src() -> AsyncIterator[bytes]:
        nonlocal closed
        try:
            yield b"data: 1\n\n"
            yield b"data: 2\n"
            yield b"\n"
        finally:
            closed = True

    async for c in passthrough_with_tap(src(), on_event=seen.append):
        out.append(c)
    assert out == [b"data: 1\n\n", b"data: 2\n", b"\n"]
    assert [e.data for e in seen] == ["1", "2"] and closed


async def test_passthrough_propagates_close() -> None:
    closed = False

    async def src() -> AsyncIterator[bytes]:
        nonlocal closed
        try:
            yield b"a"
            yield b"b"
        finally:
            closed = True

    gen = passthrough_with_tap(src())
    assert await gen.__anext__() == b"a"
    await gen.aclose()  # type: ignore[attr-defined]
    assert closed
