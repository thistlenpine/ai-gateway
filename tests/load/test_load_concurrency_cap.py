"""Per-client in-flight cap (SEC-22) and bounded stream lifetime (SEC-23) under concurrent streams."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from load.load_rig import INVOICES, HangingStream, boot_load_gateway

SSE = {"content-type": "text/event-stream"}


def _hang_route(gw: Any, streams: list[HangingStream]) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        st = HangingStream([b": hi\n\n"])
        streams.append(st)
        return httpx.Response(200, headers=SSE, stream=st)

    gw.upstream.get(INVOICES).mock(side_effect=handler)


async def test_inflight_cap_rejects_excess_streams_and_frees_slots_on_disconnect() -> None:
    async with boot_load_gateway(limits={"requests_per_window": 1000, "max_concurrent_requests": 3}) as gw:
        streams: list[HangingStream] = []
        _hang_route(gw, streams)
        tasks = [asyncio.create_task(gw.get("streamer")) for _ in range(6)]
        for _ in range(200):
            if len(streams) == 3 and sum(t.done() for t in tasks) == 3:
                break
            await asyncio.sleep(0.01)
        done = [t.result() for t in tasks if t.done()]
        assert len(streams) == 3  # only 3 reached the upstream
        assert [r.status_code for r in done] == [429, 429, 429]
        assert all(r.json()["error"]["code"] == "too_many_concurrent_requests" for r in done)
        assert all(r.headers["retry-after"] == "1" for r in done)
        # other clients are not affected by this client's open streams
        gw.upstream.get(INVOICES).respond(json={"ok": True})
        assert (await gw.get("bystander")).status_code == 200
        for t in tasks:
            t.cancel()  # all callers disconnect
        await asyncio.gather(*tasks, return_exceptions=True)
        for _ in range(50):
            await asyncio.sleep(0.01)
            if all(s.closed.is_set() for s in streams):
                break
        assert all(s.closed.is_set() for s in streams)
        assert gw.tel.active == 0
        # slots are back: the next request goes through
        _hang_route(gw, streams)
        t = asyncio.create_task(gw.get("streamer"))
        for _ in range(100):
            if len(streams) == 4:
                break
            await asyncio.sleep(0.01)
        assert len(streams) == 4
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)


async def test_stream_is_closed_cleanly_when_max_lifetime_elapses() -> None:
    async with boot_load_gateway(
        limits={"requests_per_window": 1000, "max_concurrent_requests": 2, "max_stream_seconds": 0.3},
        real_time=True,
    ) as gw:
        streams: list[HangingStream] = []
        _hang_route(gw, streams)
        t0 = asyncio.get_running_loop().time()
        r = await asyncio.wait_for(gw.get("lifetime"), 5)  # server ends the (SSE) body itself
        elapsed = asyncio.get_running_loop().time() - t0
        assert r.status_code == 200 and b": hi" in r.content
        assert 0.25 <= elapsed < 3, elapsed
        assert streams[0].closed.is_set()
        assert gw.tel.active == 0
        # the in-flight slot was released: limit is 2, run 4 sequential expiring streams
        for _ in range(4):
            assert (await asyncio.wait_for(gw.get("lifetime"), 5)).status_code == 200


async def test_concurrency_slot_store_failures_follow_on_error_policy() -> None:
    async with boot_load_gateway(limits={"requests_per_window": 1000}) as gw:
        gw.upstream.get(INVOICES).respond(json={"ok": True})
        lim = gw.services.limiter

        async def broken(*_a: Any, **_k: Any) -> Any:
            raise ConnectionError("store down")

        original_acquire, original_release = gw.services.store.acquire, gw.services.store.release
        gw.services.store.acquire = broken  # type: ignore[method-assign]
        gw.services.store.release = broken  # type: ignore[method-assign]
        assert await lim.acquire_concurrency("c", gw.services.config.limits_for("c")) == ""  # fail_open
        await lim.release_concurrency("")  # no slot held: no-op
        await lim.release_concurrency("rl:x:c:conc")  # store error swallowed
        lim._on_error = "fail_closed"
        assert await lim.acquire_concurrency("c", gw.services.config.limits_for("c")) is None
        gw.services.store.acquire, gw.services.store.release = original_acquire, original_release  # type: ignore[method-assign]
