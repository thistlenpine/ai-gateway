"""Shared helpers for auth tests."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from opentelemetry.trace import INVALID_SPAN, Span

from ai_gateway.config import IdPConfig
from ai_gateway.observability.telemetry import AttrValue, NullTelemetry

ISSUER = "https://idp.test"
JWKS_URL = "https://idp.test/jwks.json"
AUDIENCE = "ai-gateway"


class RecTelemetry(NullTelemetry):
    def __init__(self) -> None:
        super().__init__()
        self.failures: list[tuple[str, str | None]] = []
        self.jwks: list[str] = []
        self.spans: list[str] = []

    def record_auth_failure(self, reason: str, *, issuer: str | None = None) -> None:
        self.failures.append((reason, issuer))

    def record_jwks_refresh(self, *, issuer: str, outcome: str) -> None:
        self.jwks.append(outcome)

    def span(self, name: str, attributes: Any = None) -> AbstractContextManager[Span]:
        self.spans.append(name)
        return nullcontext(INVALID_SPAN)


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def make_idp(**kw: Any) -> IdPConfig:
    base: dict[str, Any] = {
        "name": "main",
        "issuer": ISSUER,
        "jwks_uri": JWKS_URL,
        "audiences": [AUDIENCE],
    }
    base.update(kw)
    return IdPConfig(**base)


__all__ = ["AttrValue", "FakeClock", "RecTelemetry", "make_idp"]
