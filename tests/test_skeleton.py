from pathlib import Path

import ai_gateway
from ai_gateway.config import load_config
from ai_gateway.models import LimitDecision, LimitScope, Phase

EXAMPLES = Path(__file__).parent.parent / "config" / "examples"


def test_import() -> None:
    assert ai_gateway.__version__


def test_example_config_loads() -> None:
    cfg = load_config(EXAMPLES / "gateway.yaml")
    assert cfg.policy.upstreams
    assert cfg.limits_for("svc-batch").requests_per_window != cfg.limits_for("nobody").requests_per_window


def test_limit_headers() -> None:
    d = LimitDecision(Phase.REJECT, LimitScope.REQUESTS, "c", 10, 0, 4.2, retry_after_s=3.1)
    assert d.headers()["Retry-After"] == "4"
