"""Pure phase math (no I/O) - unit-test exhaustively. OWNER: ratelimit."""

from __future__ import annotations

from ai_gateway.config import LimitPolicy
from ai_gateway.models import Phase


def sliding_count(current: int, previous: int, elapsed_fraction: float) -> float:
    """Sliding-window-counter estimate: current + previous * (1 - elapsed_fraction), where
    elapsed_fraction in [0,1) is how far into the current fixed bucket we are."""
    frac = min(max(elapsed_fraction, 0.0), 1.0)
    return current + previous * (1.0 - frac)


def classify(ratio: float, policy: LimitPolicy) -> Phase:
    """ratio = estimated attempts in window / quota, INCLUDING this request.
      ratio <  delay_start            -> ALLOW
      delay_start <= ratio <= reject_at -> DELAY   (ALLOW if policy.max_delay_ms == 0)
      ratio >  reject_at              -> REJECT
    BLOCK is not decided here (needs the sustained-abuse timer, see RateLimiter)."""
    if ratio > policy.reject_at:
        return Phase.REJECT
    if ratio < policy.delay_start or policy.max_delay_ms == 0:
        return Phase.ALLOW
    return Phase.DELAY


def delay_for(ratio: float, policy: LimitPolicy) -> float:
    """Seconds to wait in phase 1: linear 0 -> max_delay_ms across [delay_start, reject_at],
    always <= max_delay_ms/1000."""
    cap = policy.max_delay_ms / 1000.0
    span = policy.reject_at - policy.delay_start
    if span <= 0:
        return cap if ratio >= policy.delay_start else 0.0
    frac = min(max((ratio - policy.delay_start) / span, 0.0), 1.0)
    return min(frac * cap, cap)


def retry_after(count: float, policy: LimitPolicy) -> float:
    """Seconds until the sliding estimate would fall back to <= reject_at*quota (>= 1.0,
    <= window_seconds). Assumes the estimate decays linearly to zero over one window."""
    threshold = policy.reject_at * policy.requests_per_window
    if count <= threshold or count <= 0:
        return 1.0
    seconds = policy.window_seconds * (1.0 - threshold / count)
    return min(max(seconds, 1.0), max(policy.window_seconds, 1.0))
