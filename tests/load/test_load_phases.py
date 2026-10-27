"""Phased limiting under concurrent mixed traffic: phase 1 delays, phase 2 429s, phase 3 blocks,
per client, with no bleed between clients. Fake clock => exact, deterministic expectations."""

from __future__ import annotations

import asyncio
import os

import pytest

from load.load_rig import LoadGateway, boot_load_gateway

N_CLIENTS = 24
# quota 10, delay_start .5, reject_at 1.0, block_at 1.5 -> request k (1-based) has ratio k/10:
#   k<=4 allow, 5<=k<=10 delay (k=5 has zero delay), 11..: reject
EXPECTED_STATUS = [200] * 10 + [429] * 5
EXPECTED_DELAYED = 5  # k=6..10 actually sleep (k=5 sits exactly on delay_start => 0s)


async def hammer(gw: LoadGateway, client_id: str, n: int) -> list[int]:
    out = []
    for _ in range(n):
        out.append((await gw.get(client_id)).status_code)
    return out


@pytest.fixture(autouse=True)
def _env() -> None:
    os.environ.setdefault("E2E_MCP_TOKEN", "x")
    os.environ.setdefault("E2E_LLM_KEY", "y")


async def test_three_phases_per_client_with_no_cross_client_bleed() -> None:
    async with boot_load_gateway() as gw:
        clients = [f"abuser-{i}" for i in range(N_CLIENTS)]
        quiet = [f"quiet-{i}" for i in range(N_CLIENTS)]

        async def quiet_client(c: str) -> list[int]:
            out = []
            for _ in range(3):  # stays under delay_start
                out.append((await gw.get(c)).status_code)
                await asyncio.sleep(0)
            return out

        res = await asyncio.gather(*(hammer(gw, c, 15) for c in clients), *(quiet_client(c) for c in quiet))
        abusers, quiets = res[:N_CLIENTS], res[N_CLIENTS:]
        assert all(r == EXPECTED_STATUS for r in abusers)
        assert all(r == [200, 200, 200] for r in quiets)  # nobody else's storm touches them
        assert len(gw.sleeps) == EXPECTED_DELAYED * N_CLIENTS  # phase 1 happened, only for abusers
        assert all(0 < s <= 1.0 for s in gw.sleeps)

        # phase 3: sustained abuse (>= block_at for block_sustained_seconds) -> blocked
        gw.clock.advance(6)
        blocked = await asyncio.gather(*(gw.get(c) for c in clients))
        assert all(r.status_code == 429 and r.json()["error"]["code"] == "client_blocked" for r in blocked)
        assert all(int(r.headers["retry-after"]) >= 1 for r in blocked)
        # still isolated: quiet clients served, blocked ones short-circuit before touching upstream
        calls = gw.upstream.routes[0].call_count
        assert all(r.status_code == 200 for r in await asyncio.gather(*(gw.get(c) for c in quiet)))
        assert gw.upstream.routes[0].call_count == calls + N_CLIENTS
        await asyncio.gather(*(gw.get(c) for c in clients))
        assert gw.upstream.routes[0].call_count == calls + N_CLIENTS  # blocked: upstream untouched

        # cooldown expiry (60s) - but the sliding window still remembers the abuse for ~2 windows
        gw.clock.advance(61)
        still = await asyncio.gather(*(gw.get(c) for c in clients))
        assert all(r.status_code == 429 and r.json()["error"]["code"] != "client_blocked" for r in still)
        gw.clock.advance(120)
        again = await asyncio.gather(*(gw.get(c) for c in clients))
        assert all(r.status_code == 200 for r in again)  # history aged out, block expired
        assert gw.tel.active == 0
        assert not any(gw.slots(c) for c in clients)


async def test_reject_headers_and_phase_ordering_single_client() -> None:
    async with boot_load_gateway() as gw:
        seen = []
        for _ in range(12):
            r = await gw.get("solo")
            seen.append((r.status_code, int(r.headers["ratelimit-remaining"])))
        assert [s for s, _ in seen[:10]] == [200] * 10 and [s for s, _ in seen[10:]] == [429, 429]
        assert [rem for _, rem in seen[:10]] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        r = await gw.get("solo")
        assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
        assert 1 <= int(r.headers["retry-after"]) <= 60


async def test_concurrent_burst_same_client_counts_every_attempt_exactly_once() -> None:
    """100 simultaneous requests from one client: exactly the first 10 are admitted (the store is
    a synchronous critical section, so no lost updates / over-admission)."""
    async with boot_load_gateway() as gw:
        rs = await asyncio.gather(*(gw.get("burst") for _ in range(100)))
        assert sorted(r.status_code for r in rs).count(200) == 10
        assert sorted(r.status_code for r in rs).count(429) == 90
        assert gw.tel.active == 0


async def test_delay_concurrency_cap_escalates_excess_to_429_and_slots_drain() -> None:
    async with boot_load_gateway(limits={"requests_per_window": 100, "max_concurrent_delayed": 3}) as gw:
        # 49 free requests (ratio < .5), then 12 more land in the delay band at once
        for _ in range(49):
            assert (await gw.get("slow")).status_code == 200

        async def parked(_s: float) -> None:
            await gw.gate.wait()

        gw.services.limiter._sleep = parked
        gw_gate_tasks = [asyncio.create_task(gw.get("slow")) for _ in range(12)]
        await asyncio.sleep(0.05)
        assert gw.slots("slow") == 3  # cap holds
        done = [t for t in gw_gate_tasks if t.done()]
        # 1 request sits exactly on delay_start (zero delay, no slot) and is served at once; the 8
        # beyond the 3 waiters escalate to phase 2
        assert sorted(t.result().status_code for t in done) == [200] + [429] * 8
        gw.gate.set()
        rs = await asyncio.gather(*gw_gate_tasks)
        assert sum(r.status_code == 200 for r in rs) == 4
        assert gw.slots("slow") == 0 and gw.tel.active == 0
