"""No leaked delay slots / tasks / token reservations after disconnects and cancelled streams,
and bounded memory with many client ids."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from load.load_rig import (
    LLM_URL,
    HangingStream,
    LoadGateway,
    anthropic_json,
    boot_load_gateway,
    llm_body,
)


def _extra_tasks(before: set[asyncio.Task[object]]) -> set[asyncio.Task[object]]:
    return {t for t in asyncio.all_tasks() if t not in before and not t.done()}


async def _drain() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def test_cancelled_requests_in_delay_phase_release_every_slot() -> None:
    async with boot_load_gateway(
        limits={"requests_per_window": 100, "max_concurrent_delayed": 20}, gated_sleep=True
    ) as gw:
        for _ in range(50):
            await gw.get("c1")  # ratio reaches .5 -> every further request is in the delay band
        before = set(asyncio.all_tasks())
        tasks = [asyncio.create_task(gw.get("c1")) for _ in range(15)]
        while gw.sleeping < 15:
            await asyncio.sleep(0.001)
        assert gw.slots("c1") == 15 and gw.tel.active == 15
        for t in tasks:
            t.cancel()  # client disconnects while parked in the delay
        await asyncio.gather(*tasks, return_exceptions=True)
        await _drain()
        assert gw.slots("c1") == 0, "delay slots leaked after client disconnect"
        assert gw.tel.active == 0
        assert gw.sleeping == 0
        assert not _extra_tasks(before)
        # capacity is fully available again
        gw.gate.set()
        assert (await gw.get("c1")).status_code == 200


async def test_slot_released_when_upstream_fails_after_delay() -> None:
    async with boot_load_gateway(limits={"requests_per_window": 100}) as gw:
        gw.upstream.get("http://billing.internal/invoices").mock(side_effect=httpx.ConnectError("boom"))
        for _ in range(60):
            r = await gw.get("c2")
            assert r.status_code == 502
        assert gw.slots("c2") == 0 and gw.tel.active == 0
        assert len(gw.sleeps) > 0  # we did pass through phase 1


def _sse(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def _llm(gw: LoadGateway, client_id: str, **kw: object) -> httpx.Response:
    return await gw.client.post(
        "/llm/anthropic/v1/messages",
        headers=gw.headers(client_id),
        content=llm_body(stream=True, **kw),  # type: ignore[arg-type]
    )


async def test_cancelled_stream_closes_upstream_and_refunds_reserved_tokens() -> None:
    async with boot_load_gateway(tokens_per_window=1000) as gw:
        stream = HangingStream([b": keepalive\n\n"])  # no usage observed before the disconnect
        gw.upstream.post(LLM_URL).mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        )
        before = set(asyncio.all_tasks())
        t = asyncio.create_task(_llm(gw, "llm1"))
        await asyncio.wait_for(stream.sent.wait(), 5)
        assert gw.tok("llm1") > 0  # pre-flight reservation held while streaming
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        await _drain()
        assert stream.closed.is_set(), "upstream connection not closed on client disconnect"
        assert gw.tok("llm1") == 0, "reservation not refunded"
        assert gw.tel.active == 0 and gw.tel.completed == 1
        assert not _extra_tasks(before)


async def test_cancelled_stream_reconciles_with_observed_usage() -> None:
    async with boot_load_gateway(tokens_per_window=1000) as gw:
        stream = HangingStream(
            [
                _sse(
                    "message_start",
                    {"type": "message_start", "message": {"usage": {"input_tokens": 30, "output_tokens": 1}}},
                )
            ]
        )
        gw.upstream.post(LLM_URL).mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        )
        t = asyncio.create_task(_llm(gw, "llm2"))
        await asyncio.wait_for(stream.sent.wait(), 5)
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        await _drain()
        assert stream.closed.is_set()
        assert gw.tok("llm2") in (0, 31), gw.tok("llm2")  # refunded, or settled at what was observed
        assert gw.tok("llm2") <= 31


async def test_completed_llm_calls_settle_to_actual_usage_and_never_go_negative() -> None:
    async with boot_load_gateway(tokens_per_window=100_000, limits={"requests_per_window": 1000}) as gw:
        gw.upstream.post(LLM_URL).respond(json=anthropic_json(7, 11))
        rs = await asyncio.gather(*(_llm_json(gw, f"u{i % 3}") for i in range(60)))
        assert all(r.status_code == 200 for r in rs)
        for i in range(3):
            assert gw.tok(f"u{i}") == 20 * (7 + 11)  # 20 calls each, settled to actual usage
        assert gw.tel.active == 0


async def _llm_json(gw: LoadGateway, cid: str) -> httpx.Response:
    return await gw.client.post(
        "/llm/anthropic/v1/messages", headers=gw.headers(cid), content=llm_body(stream=False)
    )


async def test_client_disconnect_during_buffered_request_phase_does_not_leak() -> None:
    """Cancel requests at random points of their lifecycle: nothing may remain afterwards."""
    async with boot_load_gateway(
        limits={"requests_per_window": 40}, gated_sleep=True, tokens_per_window=10_000
    ) as gw:
        gw.upstream.post(LLM_URL).respond(json=anthropic_json())
        before = set(asyncio.all_tasks())
        tasks = []
        for i in range(120):
            cid = f"r{i % 6}"
            tasks.append(asyncio.create_task(_llm_json(gw, cid) if i % 2 else gw.get(cid)))
        for step in range(8):
            await asyncio.sleep(0)
            for t in tasks[step::8]:
                t.cancel()
        gw.gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _drain()
        assert gw.tel.active == 0
        assert all(gw.slots(f"r{i}") == 0 for i in range(6))
        assert all(gw.tok(f"r{i}") >= 0 for i in range(6))
        assert not _extra_tasks(before)


# --------------------------------------------------------------------------- bounded memory


@pytest.mark.parametrize("n_clients", [1500])
async def test_store_stays_bounded_with_many_client_ids(n_clients: int) -> None:
    cap = 300
    async with boot_load_gateway(max_keys=cap) as gw:
        peak = 0
        for i in range(n_clients):
            r = await gw.get(f"id-{i}")
            assert r.status_code == 200
            peak = max(peak, len(gw.services.store))  # type: ignore[arg-type]
        assert peak <= cap + 3, peak  # a single request adds a handful of keys before the sweep runs


async def test_store_expires_idle_clients() -> None:
    async with boot_load_gateway(max_keys=10_000) as gw:
        for i in range(500):
            await gw.get(f"idle-{i}")
        assert len(gw.services.store) >= 500  # type: ignore[arg-type]
        gw.clock.advance(3600)
        await gw.get("trigger-sweep")
        # lazy + periodic sweep: everything older than its TTL is gone
        for i in range(5):
            gw.clock.advance(61)
            await gw.get("trigger-sweep")
        assert len(gw.services.store) < 20  # type: ignore[arg-type]


async def test_cancelled_stream_closes_upstream_and_leaks_nothing() -> None:
    async with boot_load_gateway(tokens_per_window=1000) as gw:
        stream = HangingStream([b": keepalive\n\n"])
        gw.upstream.post(LLM_URL).mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        )
        before = set(asyncio.all_tasks())
        t = asyncio.create_task(_llm(gw, "llm3"))
        await asyncio.wait_for(stream.sent.wait(), 5)
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        await _drain()
        assert stream.closed.is_set()
        assert gw.tel.active == 0 and gw.tel.completed == 1
        assert not _extra_tasks(before)


async def test_block_survives_flood_of_other_client_ids_at_default_scale() -> None:
    """A blocked client must stay blocked while many other ids churn the store (cap not reached)."""
    async with boot_load_gateway(max_keys=5000) as gw:
        for _ in range(15):
            await gw.get("villain")
        gw.clock.advance(6)
        assert (await gw.get("villain")).json()["error"]["code"] == "client_blocked"
        for i in range(800):
            await gw.get(f"flood-{i}")
        assert (await gw.get("villain")).json()["error"]["code"] == "client_blocked"


# Regression: eviction used to drop `block` keys first (they expire soonest) once the store was full.
async def test_block_survives_flood_when_store_is_at_capacity() -> None:
    async with boot_load_gateway(max_keys=300) as gw:
        for _ in range(15):
            await gw.get("villain")
        gw.clock.advance(6)
        assert (await gw.get("villain")).json()["error"]["code"] == "client_blocked"
        for i in range(1000):
            await gw.get(f"flood-{i}")
        r = await gw.get("villain")
        assert r.status_code == 429 and r.json()["error"]["code"] == "client_blocked"
