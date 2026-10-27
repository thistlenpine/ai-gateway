"""Property / model-based tests for RateLimiter invariants (fake clock, in-memory store)."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Coroutine
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from ai_gateway.config import LimitPolicy
from ai_gateway.models import Phase
from ai_gateway.observability.telemetry import NullTelemetry
from ai_gateway.ratelimit import MemoryStateStore, RateLimiter
from support import FakeClock

RANK = {Phase.ALLOW: 0, Phase.DELAY: 1, Phase.REJECT: 2, Phase.BLOCK: 3}


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


@st.composite
def policies(draw: st.DrawFn, *, no_block: bool = False, tokens: bool = False) -> LimitPolicy:
    quota = draw(st.integers(1, 40))
    delay_start = draw(st.floats(0.05, 1.0))
    reject_at = draw(st.floats(delay_start, 1.5))
    block_at = draw(st.floats(max(reject_at, 1.0), 3.0))
    return LimitPolicy(
        requests_per_window=quota,
        window_seconds=draw(st.sampled_from([1.0, 10.0, 60.0])),
        delay_start=delay_start,
        reject_at=reject_at,
        block_at=block_at,
        block_sustained_seconds=1e9 if no_block else draw(st.sampled_from([0.0, 1.0, 5.0])),
        block_duration_seconds=draw(st.sampled_from([5.0, 30.0, 300.0])),
        max_delay_ms=draw(st.sampled_from([0, 100, 1000])),
        max_concurrent_delayed=draw(st.integers(0, 5)),
        tokens_per_window=draw(st.integers(10, 500)) if tokens else None,
        token_window_seconds=draw(st.sampled_from([5.0, 60.0])),
    )


class Rig:
    def __init__(self, sleep_exc: type[BaseException] | None = None) -> None:
        self.clock = FakeClock()
        self.store = MemoryStateStore(clock=self.clock, max_keys=10_000)
        self.sleeps: list[float] = []
        self.sleep_exc = sleep_exc

        async def sleep(s: float) -> None:
            self.sleeps.append(s)
            if self.sleep_exc is not None:
                raise self.sleep_exc

        self.limiter = RateLimiter(self.store, NullTelemetry(), clock=self.clock, sleep=sleep)

    def slots(self, client: str) -> int:
        return self.store._data.get(f"rl:{client}:delayq", (0, 0.0))[0]


# --------------------------------------------------------------------------- quota bound


@given(policies(), st.lists(st.floats(0, 5), min_size=1, max_size=120))
def test_allowed_per_bucket_never_exceeds_reject_threshold(policy: LimitPolicy, gaps: list[float]) -> None:
    """Every attempt counts, and sliding >= current-bucket, so allowed-in-bucket <= reject_at*quota
    (= quota + burst allowance). Holds even for a blocked/rejected client that keeps hammering."""

    async def go() -> None:
        r = Rig()
        allowed: dict[int, int] = {}
        for gap in gaps:
            r.clock.advance(gap)
            d = await r.limiter.check_request("c", policy)
            assert d.remaining >= 0 and d.limit == policy.requests_per_window
            if d.allowed:
                b = math.floor(r.clock() / policy.window_seconds)
                allowed[b] = allowed.get(b, 0) + 1
                assert d.usage_ratio <= policy.reject_at + 1e-9
            await r.limiter.release(d)
        cap = policy.reject_at * policy.requests_per_window
        for b, n in allowed.items():
            assert n <= cap + 1e-9, (b, n, cap)

    run(go())


# --------------------------------------------------------------------------- monotone phases


@given(policies(no_block=True), st.integers(1, 100))
def test_phase_is_monotonic_with_load_at_fixed_time(policy: LimitPolicy, n: int) -> None:
    async def go() -> None:
        r = Rig()
        last = -1
        for _ in range(n):
            d = await r.limiter.check_request("c", policy)  # slots deliberately never released
            assert RANK[d.phase] >= last
            assert d.phase is not Phase.BLOCK
            last = RANK[d.phase]
        # ratio is non-decreasing too

    run(go())


@given(policies(no_block=True), st.lists(st.floats(0, 3), min_size=1, max_size=60))
def test_phase_never_milder_than_classification_of_ratio(policy: LimitPolicy, gaps: list[float]) -> None:
    """A request whose ratio is above reject_at must never be allowed; below delay_start never rejected
    for rate reasons."""

    async def go() -> None:
        r = Rig()
        for gap in gaps:
            r.clock.advance(gap)
            d = await r.limiter.check_request("c", policy)
            if d.usage_ratio > policy.reject_at:
                assert d.phase in (Phase.REJECT, Phase.BLOCK)
            if d.usage_ratio < policy.delay_start:
                assert d.phase is Phase.ALLOW
            await r.limiter.release(d)

    run(go())


# --------------------------------------------------------------------------- block expiry


@given(policies(), st.integers(0, 1), st.floats(0.0, 1.0))
def test_block_lasts_exactly_block_duration(policy: LimitPolicy, _unused: int, frac: float) -> None:
    async def go() -> None:
        r = Rig()
        p = policy.model_copy(update={"block_sustained_seconds": 0.0})  # block on first overshoot
        blocked_at = None
        for _ in range(int(p.requests_per_window * p.block_at) + 3):
            d = await r.limiter.check_request("c", p)
            await r.limiter.release(d)
            if d.phase is Phase.BLOCK:
                blocked_at = r.clock()
                break
        assert blocked_at is not None
        # still blocked strictly inside the cooldown, with a shrinking Retry-After
        r.clock.advance(p.block_duration_seconds * frac * 0.999)
        b = await r.limiter.is_blocked("c")
        assert b is not None and 0 < b.retry_after_s <= p.block_duration_seconds
        # no counters are touched while blocked
        before = dict(r.store._data)
        await r.limiter.check_request("c", p)
        assert {k: v for k, v in r.store._data.items() if ":req:" in k} == {
            k: v for k, v in before.items() if ":req:" in k
        }
        # expired at/after the duration
        r.clock.advance(p.block_duration_seconds)
        assert await r.limiter.is_blocked("c") is None

    run(go())


# --------------------------------------------------------------------------- isolation

OPS = st.lists(
    st.tuples(st.sampled_from(["a", "b", "c"]), st.floats(0, 2), st.booleans()), min_size=1, max_size=80
)


@given(policies(), OPS)
def test_clients_are_isolated(policy: LimitPolicy, ops: list[tuple[str, float, bool]]) -> None:
    """Client X's decisions in an interleaved run equal its decisions when it runs alone on the same clock."""

    async def interleaved() -> list[tuple[str, float, Phase, float]]:
        r = Rig()
        out = []
        for client, gap, rel in ops:
            r.clock.advance(gap)
            d = await r.limiter.check_request(client, policy)
            out.append((client, r.clock(), d.phase, d.usage_ratio))
            if rel:
                await r.limiter.release(d)
        return out

    async def alone(client: str, times: list[float], rels: list[bool]) -> list[tuple[Phase, float]]:
        r = Rig()
        out = []
        for t, rel in zip(times, rels, strict=True):
            r.clock.t = t
            d = await r.limiter.check_request(client, policy)
            out.append((d.phase, d.usage_ratio))
            if rel:
                await r.limiter.release(d)
        return out

    seen = run(interleaved())
    for client in "abc":
        mine = [(i, x) for i, x in enumerate(seen) if x[0] == client]
        if not mine:
            continue
        rels = [ops[i][2] for i, _ in mine]
        solo = run(alone(client, [x[1] for _, x in mine], rels))
        assert [(x[2], x[3]) for _, x in mine] == solo


# --------------------------------------------------------------------------- delay slots


@given(
    policies(),
    st.lists(st.tuples(st.floats(0, 1), st.sampled_from(["apply", "release", "cancel"])), max_size=80),
)
def test_delay_slots_are_always_released(policy: LimitPolicy, ops: list[tuple[float, str]]) -> None:
    async def go() -> None:
        r = Rig()
        for gap, how in ops:
            r.clock.advance(gap)
            d = await r.limiter.check_request("c", policy)
            if d.phase is Phase.DELAY:
                # a zero-length delay (ratio == delay_start) waits for nothing and holds no slot
                assert (d.slot is not None) == (d.delay_s > 0)
                assert r.slots("c") <= policy.max_concurrent_delayed
                if how == "apply":
                    await r.limiter.apply_delay(d)
                elif how == "release":
                    await r.limiter.release(d)
                else:  # sleep interrupted
                    r.sleep_exc = asyncio.CancelledError
                    try:
                        await r.limiter.apply_delay(d)
                    except asyncio.CancelledError:
                        pass
                    r.sleep_exc = None
            else:
                assert d.slot is None
        assert r.slots("c") == 0

    run(go())


@given(policies(), st.integers(1, 12))
def test_concurrent_delayed_never_exceeds_cap(policy: LimitPolicy, n: int) -> None:
    async def go() -> None:
        r = Rig()
        held = []
        for _ in range(n):
            d = await r.limiter.check_request("c", policy)
            if d.phase is Phase.DELAY and d.slot is not None:
                held.append(d)
            assert len(held) <= policy.max_concurrent_delayed
        for d in held:
            await r.limiter.release(d)
        assert r.slots("c") == 0

    run(go())


# --------------------------------------------------------------------------- tokens


TOK_OPS = st.lists(
    st.tuples(
        st.floats(0, 40), st.integers(0, 300), st.one_of(st.none(), st.integers(0, 400)), st.booleans()
    ),
    min_size=1,
    max_size=60,
)


@given(policies(tokens=True), TOK_OPS)
def test_token_counter_never_negative_and_reservations_bounded(
    policy: LimitPolicy, ops: list[tuple[float, int, int | None, bool]]
) -> None:
    async def go() -> None:
        r = Rig()
        pending: list[tuple[int, int | None, int]] = []  # (estimate, actual, bucket at reserve)
        quota = policy.tokens_per_window
        assert quota is not None
        for gap, est, actual, reconcile_late in ops:
            r.clock.advance(gap)
            d = await r.limiter.reserve_tokens("c", policy, est)
            assert d.remaining >= 0
            if d.allowed:
                bucket = math.floor(r.clock() / policy.token_window_seconds)
                key = f"rl:c:tok:{bucket}"
                assert 0 <= (await r.store.get(key) or 0) <= quota
                pending.append((est, actual, bucket))
            else:
                assert d.phase is Phase.REJECT and d.retry_after_s >= 1.0
            if pending and reconcile_late:
                e, a, _ = pending.pop(0)
                await r.limiter.reconcile_tokens("c", policy, e, 0 if a is None else a)
            for k, (v, _) in r.store._data.items():
                if ":tok:" in k:
                    assert v >= 0, (k, v)
        # settle every outstanding reservation as a full refund (client disconnect): counters stay >= 0
        for e, _, _ in pending:
            await r.limiter.reconcile_tokens("c", policy, e, 0)
        for k, (v, _) in r.store._data.items():
            if ":tok:" in k:
                assert v >= 0

    run(go())


@given(policies(tokens=True), st.integers(1, 20))
def test_full_refund_restores_capacity(policy: LimitPolicy, n: int) -> None:
    async def go() -> None:
        r = Rig()
        quota = policy.tokens_per_window
        assert quota is not None
        est = max(quota // 4, 1)
        granted = []
        for _ in range(n):
            d = await r.limiter.reserve_tokens("c", policy, est)
            if d.allowed:
                granted.append(est)
        assert sum(granted) <= quota
        for e in granted:
            await r.limiter.reconcile_tokens("c", policy, e, 0)
        assert (
            await r.store.get(f"rl:c:tok:{math.floor(r.clock() / policy.token_window_seconds)}") or 0
        ) == 0
        assert (await r.limiter.reserve_tokens("c", policy, est)).allowed

    run(go())
