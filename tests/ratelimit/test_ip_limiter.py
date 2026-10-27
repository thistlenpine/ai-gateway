"""Pre-auth per-IP token bucket: rate, burst, refill, isolation, bounded memory."""

from __future__ import annotations

import pytest

from ai_gateway.config import IpLimitConfig
from ai_gateway.ratelimit import IpRateLimiter
from support import FakeClock


def _lim(**kw: object) -> tuple[IpRateLimiter, FakeClock]:
    clock = FakeClock(1000.0)
    cfg = IpLimitConfig(**{"requests_per_window": 60, "window_seconds": 60, "burst": 5, **kw})
    return IpRateLimiter(cfg, clock=clock), clock


def test_burst_then_reject_with_retry_after_then_refill() -> None:
    lim, clock = _lim()
    assert [lim.check("1.2.3.4") for _ in range(5)] == [None] * 5
    wait = lim.check("1.2.3.4")
    assert wait is not None and 0 < wait <= 1.0  # 1 token/s
    clock.advance(1.0)
    assert lim.check("1.2.3.4") is None
    assert lim.check("1.2.3.4") is not None
    assert IpRateLimiter.retry_after_header(0.2) == "1" and IpRateLimiter.retry_after_header(2.1) == "3"


def test_rejections_do_not_extend_the_penalty() -> None:
    lim, clock = _lim()
    for _ in range(5):
        lim.check("9.9.9.9")
    for _ in range(1000):  # a flood while limited must not push recovery further away
        assert lim.check("9.9.9.9") is not None
    clock.advance(1.0)
    assert lim.check("9.9.9.9") is None


def test_ips_are_isolated_and_ipv6_collapses_to_slash_64() -> None:
    lim, _ = _lim()
    for _ in range(5):
        lim.check("1.1.1.1")
    assert lim.check("1.1.1.1") is not None
    assert lim.check("2.2.2.2") is None
    for i in range(5):
        assert lim.check(f"2001:db8:1:2::{i:x}") is None  # same /64: one bucket
    assert lim.check("2001:db8:1:2:ffff::1") is not None
    assert lim.check("2001:db8:1:3::1") is None  # other /64
    assert lim.check("::ffff:1.1.1.1") is not None  # v4-mapped == the v4 address


def test_default_burst_is_requests_per_window_and_disabled_never_limits() -> None:
    lim = IpRateLimiter(IpLimitConfig(requests_per_window=50), clock=FakeClock())
    assert all(lim.check("1.1.1.1") is None for _ in range(50))
    assert lim.check("1.1.1.1") is not None
    off = IpRateLimiter(IpLimitConfig(enabled=False, requests_per_window=1), clock=FakeClock())
    assert not off.enabled and all(off.check("1.1.1.1") is None for _ in range(100))


def test_memory_is_bounded_under_distinct_ip_flood() -> None:
    lim, _ = _lim(max_tracked_ips=1000)
    for i in range(20_000):
        lim.check(f"10.{i >> 16 & 255}.{i >> 8 & 255}.{i & 255}")
        assert len(lim) <= 1000
    # idle (fully refilled) entries are swept first, so an actively limited IP keeps its state
    lim2, clock2 = _lim(max_tracked_ips=100)
    for i in range(90):
        lim2.check(f"172.16.0.{i}")
    clock2.advance(10)  # those buckets are full again: indistinguishable from new
    for _ in range(5):
        lim2.check("8.8.8.8")
    for i in range(20):
        lim2.check(f"172.17.0.{i}")
    assert len(lim2) <= 100
    assert lim2.check("8.8.8.8") is not None


def test_unparseable_ip_does_not_crash_or_share_with_others() -> None:
    lim, _ = _lim()
    for _ in range(5):
        lim.check("")
    assert lim.check("") is not None
    assert lim.check("1.2.3.4") is None


@pytest.mark.parametrize("bad", [{"requests_per_window": 0}, {"window_seconds": 0}, {"burst": 0}])
def test_config_validation(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        IpLimitConfig(**bad)
