from __future__ import annotations

import pytest

from ai_gateway.config import LimitPolicy
from ai_gateway.models import Phase
from ai_gateway.ratelimit import phases

P = LimitPolicy(requests_per_window=100, window_seconds=60, max_delay_ms=2000)


def test_sliding_count() -> None:
    assert phases.sliding_count(10, 20, 0.0) == 30
    assert phases.sliding_count(10, 20, 0.5) == 20
    assert phases.sliding_count(10, 20, 1.0) == 10
    assert phases.sliding_count(10, 20, 7.0) == 10  # clamped
    assert phases.sliding_count(10, 20, -1.0) == 30


@pytest.mark.parametrize(
    ("ratio", "phase"),
    [
        (0.0, Phase.ALLOW),
        (0.79, Phase.ALLOW),
        (0.8, Phase.DELAY),
        (1.0, Phase.DELAY),
        (1.01, Phase.REJECT),
        (5.0, Phase.REJECT),
    ],
)
def test_classify(ratio: float, phase: Phase) -> None:
    assert phases.classify(ratio, P) is phase


def test_classify_delay_disabled() -> None:
    p = P.model_copy(update={"max_delay_ms": 0})
    assert phases.classify(0.9, p) is Phase.ALLOW
    assert phases.classify(1.1, p) is Phase.REJECT


def test_delay_for_linear_and_capped() -> None:
    assert phases.delay_for(0.8, P) == 0.0
    assert phases.delay_for(0.9, P) == pytest.approx(1.0)
    assert phases.delay_for(1.0, P) == pytest.approx(2.0)
    assert phases.delay_for(9.0, P) == pytest.approx(2.0)
    assert phases.delay_for(0.1, P) == 0.0


def test_delay_for_degenerate_span() -> None:
    p = LimitPolicy(delay_start=1.0, reject_at=1.0, max_delay_ms=500)
    assert phases.delay_for(1.0, p) == 0.5
    assert phases.delay_for(0.5, p) == 0.0


def test_retry_after_bounds() -> None:
    assert phases.retry_after(50, P) == 1.0
    assert phases.retry_after(200, P) == pytest.approx(30.0)
    assert phases.retry_after(1e9, P) <= 60.0
    assert phases.retry_after(101, P) >= 1.0
