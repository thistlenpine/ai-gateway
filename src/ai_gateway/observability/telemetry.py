"""Telemetry facade - THE ONLY observability API other modules may use. OWNER: observability.

Every module receives a `Telemetry` via constructor injection (see services.py). Modules never
import opentelemetry directly (exception: proxy/ may use `opentelemetry.propagate.inject`).
`NullTelemetry` is fully implemented (no-ops) so every owner can unit-test without OTel setup.

Logging is plain stdlib: `logging.getLogger(__name__).info("msg", extra={"event": "x.y", ...})`.
observability.logging installs a JSON formatter that adds trace_id/span_id/request_id/client_id
from the active span + context.get_ctx(). Never pass tokens/headers dicts in `extra`.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol, runtime_checkable

from opentelemetry.trace import INVALID_SPAN, Span

from ai_gateway.models import LimitDecision, PolicyDecision, RequestContext, UsageRecord
from ai_gateway.observability.audit import AuditEvent

AttrValue = str | int | float | bool


@runtime_checkable
class Telemetry(Protocol):
    def span(
        self, name: str, attributes: Mapping[str, AttrValue] | None = None
    ) -> AbstractContextManager[Span]:
        """Internal child span of the current span. Records exceptions + sets ERROR status."""
        ...

    def add_active(self, delta: int, *, upstream: str | None = None) -> None:
        """gateway.requests.active +1 / -1."""
        ...

    def record_request(self, ctx: RequestContext) -> None:
        """gateway.requests + gateway.request.duration from the finished ctx (status_code set)."""
        ...

    def record_auth_failure(self, reason: str, *, issuer: str | None = None) -> None: ...

    def record_authz(self, decision: PolicyDecision, ctx: RequestContext) -> None:
        """gateway.authz.decisions metric (the audit event is emitted separately via audit())."""
        ...

    def record_rate_limit(self, decision: LimitDecision, ctx: RequestContext | None = None) -> None:
        """gateway.ratelimit.decisions by phase+scope; delay histogram when phase is DELAY."""
        ...

    def record_upstream(
        self,
        *,
        upstream: str,
        kind: str,
        duration_s: float,
        status: int | None,
        error_type: str | None = None,
    ) -> None:
        """gateway.upstream.duration; also gateway.upstream.errors when error_type/5xx."""
        ...

    def record_llm_tokens(self, usage: UsageRecord, *, upstream: str, client_label: str) -> None: ...

    def record_jwks_refresh(self, *, issuer: str, outcome: str) -> None: ...

    def audit(self, event: AuditEvent) -> None: ...

    async def shutdown(self) -> None:
        """Flush + stop exporters."""
        ...


class NullTelemetry:
    """No-op Telemetry for tests/dev. Collects audit events in `.audit_events` for assertions."""

    def __init__(self) -> None:
        self.audit_events: list[AuditEvent] = []

    def span(
        self, name: str, attributes: Mapping[str, AttrValue] | None = None
    ) -> AbstractContextManager[Span]:
        return nullcontext(INVALID_SPAN)

    def add_active(self, delta: int, *, upstream: str | None = None) -> None:
        return None

    def record_request(self, ctx: RequestContext) -> None:
        return None

    def record_auth_failure(self, reason: str, *, issuer: str | None = None) -> None:
        return None

    def record_authz(self, decision: PolicyDecision, ctx: RequestContext) -> None:
        return None

    def record_rate_limit(self, decision: LimitDecision, ctx: RequestContext | None = None) -> None:
        return None

    def record_upstream(
        self,
        *,
        upstream: str,
        kind: str,
        duration_s: float,
        status: int | None,
        error_type: str | None = None,
    ) -> None:
        return None

    def record_llm_tokens(self, usage: UsageRecord, *, upstream: str, client_label: str) -> None:
        return None

    def record_jwks_refresh(self, *, issuer: str, outcome: str) -> None:
        return None

    def audit(self, event: AuditEvent) -> None:
        self.audit_events.append(event)

    async def shutdown(self) -> None:
        return None
