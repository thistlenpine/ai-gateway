"""Small real-time smoke: real clock, real asyncio.sleep, many concurrent clients, mixed traffic."""

from __future__ import annotations

import asyncio
import time
from collections import Counter

import pytest

from load.load_rig import LLM_URL, LoadGateway, anthropic_json, boot_load_gateway, llm_body

pytestmark = pytest.mark.slow

LIMITS = {
    "requests_per_window": 20,
    "window_seconds": 2,
    "delay_start": 0.5,
    "reject_at": 1.0,
    "block_at": 1.5,
    "block_sustained_seconds": 0.4,
    "block_duration_seconds": 1.5,
    "max_delay_ms": 60,
    "max_concurrent_delayed": 4,
}


async def _mixed(gw: LoadGateway, cid: str, rounds: int) -> Counter[int]:
    out: Counter[int] = Counter()
    for i in range(rounds):
        if i % 4 == 3:
            r = await gw.client.post(
                "/llm/anthropic/v1/messages", headers=gw.headers(cid), content=llm_body(max_tokens=10)
            )
        else:
            r = await gw.get(cid)
        out[r.status_code] += 1
    return out


async def test_realtime_mixed_traffic_smoke() -> None:
    async with boot_load_gateway(limits=LIMITS, real_time=True, tokens_per_window=100_000) as gw:
        gw.upstream.post(LLM_URL).respond(json=anthropic_json(5, 5))
        abusers = [f"abuse-{i}" for i in range(10)]
        polite = [f"polite-{i}" for i in range(10)]

        async def polite_client(cid: str) -> Counter[int]:
            out: Counter[int] = Counter()
            for _ in range(5):  # 5 requests over ~0.5s: far below delay_start (10)
                out[(await gw.get(cid)).status_code] += 1
                await asyncio.sleep(0.1)
            return out

        t0 = time.monotonic()
        results = await asyncio.gather(
            *(_mixed(gw, c, 120) for c in abusers), *(polite_client(c) for c in polite)
        )
        elapsed = time.monotonic() - t0
        abuse_res, polite_res = results[:10], results[10:]
        for res in polite_res:
            assert res == Counter({200: 5}), res  # zero bleed from the storm next door
        for res in abuse_res:
            assert set(res) <= {200, 429}, res  # never a 5xx
            assert res[429] > 0  # throttled (phase 2 and/or 3)
            # cap: at most reject_at*quota per fixed 2s bucket
            buckets = int(elapsed // 2) + 2
            assert res[200] <= 20 * buckets, (res, elapsed)
        assert gw.sleeps and max(gw.sleeps) <= 0.06 + 1e-9  # phase 1 delays happened, capped
        assert gw.tel.active == 0
        assert not any(gw.slots(c) for c in abusers + polite)

        # at least some abusers reached phase 3
        codes = []
        for c in abusers:
            r = await gw.get(c)
            codes.append(r.json().get("error", {}).get("code"))
        assert "client_blocked" in codes
        # ...and the block really expires in real time
        await asyncio.sleep(LIMITS["block_duration_seconds"] + 2.2)
        assert (await gw.get(abusers[0])).status_code == 200
