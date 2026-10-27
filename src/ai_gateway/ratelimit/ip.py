"""Pre-authentication per-IP limiter. OWNER: ratelimit.

A deliberately cheap token bucket (no store round trips, no JWT work) that runs BEFORE token
validation so unauthenticated floods cannot burn CPU on signature checks or audit/log volume.
State is in-process and bounded: IPv6 addresses collapse to their /64, entries live in an LRU
`OrderedDict` capped at `max_tracked_ips`, and a bucket that has refilled completely is
indistinguishable from a new one, so it is dropped on the next sweep. Evicting an entry only ever
*resets* that IP to a full bucket (fail-open), never penalises anyone.

This is a backstop, not a WAF: keep an L7 limiter in front in production. Like every limiter in
this process it is per instance: N replicas allow N times the rate.
"""

from __future__ import annotations

import ipaddress
import math
import time
from collections import OrderedDict
from collections.abc import Callable

from ai_gateway.config import IpLimitConfig


class IpRateLimiter:
    def __init__(self, cfg: IpLimitConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._cfg = cfg
        self._clock = clock
        self._capacity = float(cfg.burst or cfg.requests_per_window)
        self._rate = cfg.requests_per_window / cfg.window_seconds  # tokens per second
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # key -> (tokens, ts)

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def __len__(self) -> int:
        return len(self._buckets)

    @staticmethod
    def _key(ip: str) -> str:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return ip[:64] or "unknown"
        if isinstance(addr, ipaddress.IPv6Address):
            if addr.ipv4_mapped is not None:
                return str(addr.ipv4_mapped)
            return str(ipaddress.ip_network(f"{addr}/64", strict=False).network_address) + "/64"
        return str(addr)

    def check(self, ip: str) -> float | None:
        """Consume one token for `ip`. None if allowed, else seconds until a token is available."""
        if not self._cfg.enabled:
            return None
        now = self._clock()
        key = self._key(ip)
        tokens, ts = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + max(now - ts, 0.0) * self._rate)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            allowed = True
            wait = 0.0
        else:
            self._buckets[key] = (tokens, now)
            allowed = False
            wait = (1.0 - tokens) / self._rate
        self._buckets.move_to_end(key)
        if len(self._buckets) > self._cfg.max_tracked_ips:
            self._evict(now)
        return None if allowed else max(wait, 0.001)

    def _evict(self, now: float) -> None:
        """Drop fully-refilled (idle) entries first, then least-recently-used ones to 90% of cap."""
        for key in [
            k for k, (t, ts) in self._buckets.items() if t + (now - ts) * self._rate >= self._capacity
        ]:
            del self._buckets[key]
        target = int(self._cfg.max_tracked_ips * 0.9)
        while len(self._buckets) > target:
            self._buckets.popitem(last=False)

    @staticmethod
    def retry_after_header(wait_s: float) -> str:
        return str(max(math.ceil(wait_s), 1))
