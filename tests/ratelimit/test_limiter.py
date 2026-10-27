from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from ai_gateway.config import LimitPolicy
from ai_gateway.models import LimitDecision, LimitScope, Phase
from ai_gateway.observability.audit import AuditType
from ai_gateway.observability.telemetry import NullTelemetry
from ai_gateway.ratelimit import MemoryStateStore, RateLimiter


class FakeClock:
    def __init__(self, t: float = 6000.0) -> None:  # bucket-aligned for 60 s windows
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class RecTelemetry(NullTelemetry):
    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[LimitDecision] = []

    def record_rate_limit(self, decision: LimitDecision, ctx: object = None) -> None:
        self.decisions.append(decision)

    def count(self, phase: Phase, scope: LimitScope = LimitScope.REQUESTS) -> int:
        return sum(1 for d in self.decisions if d.phase is phase and d.scope is scope)


def make(
    on_error: str = "fail_open",
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> tuple[RateLimiter, FakeClock, RecTelemetry, MemoryStateStore]:
    clock = FakeClock()
    store = MemoryStateStore(clock=clock)
    tel = RecTelemetry()

    async def fake_sleep(s: float) -> None:
        clock.advance(s)

    lim = RateLimiter(store, tel, clock=clock, sleep=sleep or fake_sleep, on_error=on_error)
    return lim, clock, tel, store


# quota 10: ALLOW <8, DELAY 8..10 (up to 1s), REJECT >10, BLOCK >=15 sustained 5s for 30s
POL = LimitPolicy(
    requests_per_window=10,
    window_seconds=60,
    delay_start=0.8,
    reject_at=1.0,
    block_at=1.5,
    block_sustained_seconds=5,
    block_duration_seconds=30,
    max_delay_ms=1000,
    max_concurrent_delayed=2,
)


async def test_phase_transitions_allow_delay_reject() -> None:
    lim, _, tel, _ = make()
    phases = []
    for _ in range(12):
        d = await lim.check_request("a", POL)
        phases.append(d.phase)
        await lim.apply_delay(d)
    assert phases == [Phase.ALLOW] * 7 + [Phase.DELAY] * 3 + [Phase.REJECT] * 2
    assert tel.count(Phase.REJECT) == 2
    assert tel.count(Phase.DELAY) == 3


async def test_delay_is_linear_and_slept() -> None:
    lim, clock, _, _ = make()
    for _ in range(7):
        await lim.check_request("a", POL)
    d8 = await lim.check_request("a", POL)  # ratio .8
    assert d8.phase is Phase.DELAY and d8.delay_s == 0.0 and d8.slot is None
    d9 = await lim.check_request("a", POL)  # ratio .9
    assert d9.delay_s == pytest.approx(0.5)
    d10 = await lim.check_request("a", POL)  # ratio 1.0
    assert d10.delay_s == pytest.approx(1.0)
    t0 = clock.t
    waited = await lim.apply_delay(d10)
    assert waited == pytest.approx(1.0)
    assert clock.t - t0 == pytest.approx(1.0)


async def test_reject_headers_and_rejected_counts() -> None:
    lim, _, _, _ = make()
    for _ in range(10):
        await lim.check_request("a", POL)
    d = await lim.check_request("a", POL)
    assert d.phase is Phase.REJECT and not d.allowed
    h = d.headers()
    assert h["RateLimit-Limit"] == "10" and h["RateLimit-Remaining"] == "0"
    assert int(h["Retry-After"]) >= 1
    assert h["RateLimit-Policy"] == "10;w=60"
    d2 = await lim.check_request("a", POL)
    assert d2.usage_ratio > d.usage_ratio  # rejected attempts count


async def test_max_delay_zero_disables_delay_phase() -> None:
    lim, _, _, _ = make()
    p = POL.model_copy(update={"max_delay_ms": 0})
    ds = [await lim.check_request("a", p) for _ in range(11)]
    assert [d.phase for d in ds] == [Phase.ALLOW] * 10 + [Phase.REJECT]


async def test_delay_concurrency_cap_escalates_to_reject() -> None:
    release = asyncio.Event()

    async def blocking_sleep(_: float) -> None:
        await release.wait()

    lim, _, _, store = make(sleep=blocking_sleep)
    p = POL.model_copy(update={"requests_per_window": 100, "max_concurrent_delayed": 2})
    for _ in range(80):
        await lim.check_request("a", p)
    d1 = await lim.check_request("a", p)
    d2 = await lim.check_request("a", p)
    d3 = await lim.check_request("a", p)
    assert d1.phase is Phase.DELAY and d1.slot
    assert d2.phase is Phase.DELAY
    assert d3.phase is Phase.REJECT and d3.reason == "delay_capacity"
    tasks = [asyncio.create_task(lim.apply_delay(d)) for d in (d1, d2)]
    await asyncio.sleep(0)
    d4 = await lim.check_request("a", p)
    assert d4.phase is Phase.REJECT  # slots still held
    release.set()
    await asyncio.gather(*tasks)
    d5 = await lim.check_request("a", p)
    assert d5.phase is Phase.DELAY  # slots freed
    assert await store.get("rl:a:delayq") == 1


async def test_apply_delay_cancel_releases_slot() -> None:
    async def forever(_: float) -> None:
        await asyncio.Event().wait()

    lim, _, _, store = make(sleep=forever)
    for _ in range(8):
        await lim.check_request("a", POL)
    d = await lim.check_request("a", POL)
    assert d.slot and d.delay_s > 0
    task = asyncio.create_task(lim.apply_delay(d))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await store.get("rl:a:delayq") is None


async def test_release_without_waiting_and_noop_cases() -> None:
    lim, _, _, store = make()
    for _ in range(8):
        await lim.check_request("a", POL)
    d = await lim.check_request("a", POL)
    assert await store.get("rl:a:delayq") == 1
    await lim.release(d)
    assert await store.get("rl:a:delayq") is None
    allow = await lim.check_request("b", POL)
    assert await lim.apply_delay(allow) == 0.0
    await lim.release(allow)


async def test_max_wait_cap_applies() -> None:
    clock = FakeClock()
    lim = RateLimiter(
        MemoryStateStore(clock=clock),
        NullTelemetry(),
        clock=clock,
        sleep=lambda s: _advance(clock, s),
        max_wait_s=0.25,
    )
    for _ in range(9):
        await lim.check_request("a", POL)
    d = await lim.check_request("a", POL)
    assert d.delay_s == pytest.approx(0.25)


async def _advance(clock: FakeClock, s: float) -> None:
    clock.advance(s)


async def test_block_after_sustained_abuse_then_expiry() -> None:
    lim, clock, tel, store = make()
    # Burst to >= 1.5x quota (15 attempts), starts abuse timer.
    for _ in range(15):
        d = await lim.check_request("a", POL)
    assert d.phase is Phase.REJECT
    assert await lim.is_blocked("a") is None
    clock.advance(4)  # not yet sustained (5 s)
    d = await lim.check_request("a", POL)
    assert d.phase is Phase.REJECT
    clock.advance(1.5)
    d = await lim.check_request("a", POL)
    assert d.phase is Phase.BLOCK and d.reason == "sustained_abuse"
    assert d.retry_after_s == 30
    assert d.headers()["Retry-After"] == "30"
    assert [e.type for e in tel.audit_events].count(AuditType.CLIENT_BLOCKED) == 1

    # Fast path: no counter writes while blocked.
    keys_before = dict(store._data)
    clock.advance(10)
    b = await lim.is_blocked("a")
    assert b is not None and b.phase is Phase.BLOCK
    assert b.retry_after_s == pytest.approx(20.0)
    d = await lim.check_request("a", POL)
    assert d.phase is Phase.BLOCK
    assert store._data == keys_before

    # Cooldown expiry (30 s from block at t+5.5): after 20+ more seconds.
    clock.advance(25)
    assert await lim.is_blocked("a") is None
    d = await lim.check_request("a", POL)
    assert d.phase is not Phase.BLOCK


async def test_block_immediate_when_sustained_zero() -> None:
    lim, _, _, _ = make()
    p = POL.model_copy(update={"block_sustained_seconds": 0})
    last = None
    for _ in range(15):
        last = await lim.check_request("a", p)
    assert last is not None and last.phase is Phase.BLOCK


async def test_abuse_timer_reset_when_ratio_drops() -> None:
    lim, clock, _, _ = make()
    for _ in range(15):
        await lim.check_request("a", POL)
    clock.advance(130)  # windows fully rolled -> ratio low, timer cleared
    d = await lim.check_request("a", POL)
    assert d.phase is Phase.ALLOW
    for _ in range(14):
        d = await lim.check_request("a", POL)
    assert d.phase is Phase.REJECT  # new timer starts now, not blocked
    clock.advance(3)
    d = await lim.check_request("a", POL)
    assert d.phase is Phase.REJECT


async def test_window_rolls_over() -> None:
    lim, clock, _, _ = make()
    for _ in range(11):
        d = await lim.check_request("a", POL)
    assert d.phase is Phase.REJECT
    clock.advance(130)
    d = await lim.check_request("a", POL)
    assert d.phase is Phase.ALLOW and d.remaining == 9


async def test_sliding_window_weights_previous_bucket() -> None:
    lim, clock, _, _ = make()
    for _ in range(10):
        await lim.check_request("a", POL)
    clock.advance(60 + 30)  # halfway through next bucket: prev 10 * .5 = 5
    d = await lim.check_request("a", POL)
    assert d.usage_ratio == pytest.approx(0.6)
    assert d.phase is Phase.ALLOW


async def test_client_isolation() -> None:
    lim, _, _, _ = make()
    for _ in range(15):
        await lim.check_request("noisy", POL)
    p2 = POL.model_copy(update={"block_sustained_seconds": 0})
    for _ in range(15):
        await lim.check_request("noisy2", p2)
    assert (await lim.is_blocked("noisy2")) is not None
    assert await lim.is_blocked("quiet") is None
    d = await lim.check_request("quiet", POL)
    assert d.phase is Phase.ALLOW and d.remaining == 9
    t = await lim.reserve_tokens("quiet", POL.model_copy(update={"tokens_per_window": 100}), 50)
    assert t.remaining == 50


async def test_concurrent_callers_exact_quota() -> None:
    lim, _, _, _ = make()
    p = POL.model_copy(update={"max_delay_ms": 0, "requests_per_window": 20})
    ds = await asyncio.gather(*(lim.check_request("a", p) for _ in range(50)))
    assert sum(d.phase is Phase.ALLOW for d in ds) == 20
    assert sum(d.phase is Phase.REJECT for d in ds) == 30


async def test_concurrent_delay_slots_never_exceed_cap() -> None:
    lim, _, _, store = make(sleep=lambda _: asyncio.sleep(0.01))
    p = POL.model_copy(
        update={"requests_per_window": 100, "delay_start": 0.0001, "max_concurrent_delayed": 3}
    )

    async def one() -> Phase:
        d = await lim.check_request("a", p)
        assert (await store.get("rl:a:delayq") or 0) <= 3
        await lim.apply_delay(d)
        return d.phase

    res = await asyncio.gather(*(one() for _ in range(30)))
    assert Phase.DELAY in res and Phase.REJECT in res
    assert await store.get("rl:a:delayq") is None


# ------------------------------------------------------------------ tokens

TPOL = POL.model_copy(update={"tokens_per_window": 1000, "token_window_seconds": 60})


async def test_tokens_unlimited_when_unset() -> None:
    lim, _, _, store = make()
    d = await lim.reserve_tokens("a", POL, 10**9)
    assert d.phase is Phase.ALLOW and d.scope is LimitScope.TOKENS
    assert len(store) == 0
    await lim.reconcile_tokens("a", POL, 5, 10)
    assert len(store) == 0


async def test_token_reserve_reject_rollback_and_headers() -> None:
    lim, _, tel, store = make()
    d = await lim.reserve_tokens("a", TPOL, 600)
    assert d.phase is Phase.ALLOW and d.remaining == 400
    r = await lim.reserve_tokens("a", TPOL, 500)
    assert r.phase is Phase.REJECT and r.scope is LimitScope.TOKENS
    assert r.reason == "token_quota_exceeded" and int(r.headers()["Retry-After"]) >= 1
    assert await store.get("rl:a:tok:100") == 600  # rolled back
    ok = await lim.reserve_tokens("a", TPOL, 400)
    assert ok.phase is Phase.ALLOW and ok.remaining == 0
    assert tel.count(Phase.REJECT, LimitScope.TOKENS) == 1


async def test_token_reconcile_refund_and_overage() -> None:
    lim, _, _, store = make()
    await lim.reserve_tokens("a", TPOL, 800)
    await lim.reconcile_tokens("a", TPOL, 800, 300)
    assert await store.get("rl:a:tok:100") == 300
    await lim.reconcile_tokens("a", TPOL, 100, 400)  # under-estimated: charge more
    assert await store.get("rl:a:tok:100") == 600
    await lim.reconcile_tokens("a", TPOL, 600, 0)  # aborted: refund all
    assert await store.get("rl:a:tok:100") == 0
    await lim.reconcile_tokens("a", TPOL, 50, 50)  # no-op


async def test_token_reconcile_never_negative_after_window_roll() -> None:
    lim, clock, _, store = make()
    await lim.reserve_tokens("a", TPOL, 500)
    clock.advance(61)  # next window; refund lands there
    await lim.reconcile_tokens("a", TPOL, 500, 0)
    assert await store.get("rl:a:tok:101") == 0
    d = await lim.reserve_tokens("a", TPOL, 1000)
    assert d.phase is Phase.ALLOW  # exactly the quota, no phantom capacity
    d = await lim.reserve_tokens("a", TPOL, 1)
    assert d.phase is Phase.REJECT


async def test_token_window_resets() -> None:
    lim, clock, _, _ = make()
    await lim.reserve_tokens("a", TPOL, 1000)
    assert (await lim.reserve_tokens("a", TPOL, 1)).phase is Phase.REJECT
    clock.advance(60)
    assert (await lim.reserve_tokens("a", TPOL, 1000)).phase is Phase.ALLOW


async def test_token_scope_never_blocks_or_delays() -> None:
    lim, _, _, _ = make()
    for _ in range(50):
        d = await lim.reserve_tokens("a", TPOL, 900)
        assert d.phase in (Phase.ALLOW, Phase.REJECT)
    assert await lim.is_blocked("a") is None


async def test_concurrent_token_reservations_do_not_overspend() -> None:
    lim, _, _, store = make()
    ds = await asyncio.gather(*(lim.reserve_tokens("a", TPOL, 100) for _ in range(30)))
    allowed = sum(d.phase is Phase.ALLOW for d in ds)
    assert allowed <= 10
    assert 0 <= (await store.get("rl:a:tok:100") or 0) <= 1000


# ------------------------------------------------------------------ errors / audit


class BrokenStore(MemoryStateStore):
    async def incr(self, key: str, amount: int = 1, *, ttl_s: float) -> int:
        raise RuntimeError("down")

    async def ttl(self, key: str) -> float | None:
        raise RuntimeError("down")


@pytest.mark.parametrize(("mode", "phase"), [("fail_open", Phase.ALLOW), ("fail_closed", Phase.REJECT)])
async def test_store_errors(mode: str, phase: Phase) -> None:
    tel = RecTelemetry()
    lim = RateLimiter(BrokenStore(), tel, on_error=mode)
    d = await lim.check_request("a", POL)
    assert d.phase is phase and d.reason == "limiter_error"
    t = await lim.reserve_tokens("a", TPOL, 5)
    assert t.phase is phase
    await lim.reconcile_tokens("a", TPOL, 5, 1)  # swallowed
    b = await lim.is_blocked("a")
    assert (b is None) == (mode == "fail_open")


async def test_audit_throttled_per_client_phase() -> None:
    lim, clock, tel, _ = make()
    for _ in range(14):
        await lim.check_request("a", POL)  # 4 rejects
    rejects = [e for e in tel.audit_events if e.decision == "reject"]
    assert len(rejects) == 1
    delays = [e for e in tel.audit_events if e.decision == "delay"]
    assert len(delays) == 1
    clock.advance(11)
    await lim.check_request("a", POL)
    assert len([e for e in tel.audit_events if e.decision == "reject"]) == 2


async def test_telemetry_failure_does_not_break() -> None:
    class Boom(NullTelemetry):
        def record_rate_limit(self, decision: LimitDecision, ctx: object = None) -> None:
            raise RuntimeError("x")

        def audit(self, event: object) -> None:
            raise RuntimeError("x")

    clock = FakeClock()
    lim = RateLimiter(MemoryStateStore(clock=clock), Boom(), clock=clock)
    for _ in range(12):
        d = await lim.check_request("a", POL)
    assert d.phase is Phase.REJECT
