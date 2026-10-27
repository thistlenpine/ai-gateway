from __future__ import annotations

import asyncio

import pytest

from ai_gateway.ratelimit.store import MemoryStateStore, StateStore


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


async def test_protocol_conformance() -> None:
    assert isinstance(MemoryStateStore(), StateStore)


async def test_incr_ttl_only_on_create() -> None:
    c = FakeClock()
    s = MemoryStateStore(clock=c)
    assert await s.incr("k", ttl_s=10) == 1
    c.advance(6)
    assert await s.incr("k", 4, ttl_s=10) == 5
    assert await s.ttl("k") == pytest.approx(4)
    c.advance(5)
    assert await s.get("k") is None
    assert await s.incr("k", ttl_s=10) == 1


async def test_set_get_delete_set_if_absent() -> None:
    c = FakeClock()
    s = MemoryStateStore(clock=c)
    assert await s.set_if_absent("a", 1, ttl_s=5)
    assert not await s.set_if_absent("a", 2, ttl_s=5)
    assert await s.get("a") == 1
    await s.set("a", 9, ttl_s=5)
    assert await s.get("a") == 9
    await s.delete("a")
    assert await s.ttl("a") is None
    await s.set("b", 1, ttl_s=1)
    c.advance(1)
    assert await s.set_if_absent("b", 2, ttl_s=5)


async def test_acquire_release() -> None:
    s = MemoryStateStore(clock=FakeClock())
    assert await s.acquire("q", 2, ttl_s=10)
    assert await s.acquire("q", 2, ttl_s=10)
    assert not await s.acquire("q", 2, ttl_s=10)
    await s.release("q")
    assert await s.acquire("q", 2, ttl_s=10)
    await s.release("q")
    await s.release("q")
    await s.release("q")  # floor 0, no error
    assert await s.get("q") is None
    assert not await s.acquire("z", 0, ttl_s=10)


async def test_acquire_leak_guard_ttl() -> None:
    c = FakeClock()
    s = MemoryStateStore(clock=c)
    assert await s.acquire("q", 1, ttl_s=5)
    c.advance(6)
    assert await s.acquire("q", 1, ttl_s=5)  # leaked slot expired


async def test_concurrent_incr_is_atomic() -> None:
    s = MemoryStateStore(clock=FakeClock())
    res = await asyncio.gather(*(s.incr("k", ttl_s=10) for _ in range(200)))
    assert sorted(res) == list(range(1, 201))


async def test_concurrent_acquire_respects_limit() -> None:
    s = MemoryStateStore(clock=FakeClock())
    res = await asyncio.gather(*(s.acquire("q", 7, ttl_s=10) for _ in range(50)))
    assert sum(res) == 7


async def test_eviction_drops_expired_then_idlest() -> None:
    c = FakeClock()
    s = MemoryStateStore(clock=c, max_keys=10, sweep_interval_s=1e9)
    for i in range(5):
        await s.set(f"short{i}", 1, ttl_s=1)
    c.advance(2)
    for i in range(10):
        await s.set(f"long{i}", 1, ttl_s=100 + i)
    assert len(s) <= 10
    for i in range(5):
        assert await s.get(f"short{i}") is None
    await s.set("extra", 1, ttl_s=500)  # 11th live key -> evict idlest down to 90%
    assert len(s) <= 10
    assert await s.get("extra") == 1  # newest survives
    assert await s.get("long0") is None  # closest to expiry evicted


async def test_periodic_sweep_frees_idle_keys() -> None:
    c = FakeClock()
    s = MemoryStateStore(clock=c, max_keys=1000, sweep_interval_s=10)
    for i in range(50):
        await s.set(f"k{i}", 1, ttl_s=5)
    c.advance(11)
    await s.set("fresh", 1, ttl_s=5)
    assert len(s) == 1


async def test_aclose_clears_and_bad_cap() -> None:
    s = MemoryStateStore()
    await s.set("a", 1, ttl_s=5)
    await s.aclose()
    assert len(s) == 0
    with pytest.raises(ValueError):
        MemoryStateStore(max_keys=0)
