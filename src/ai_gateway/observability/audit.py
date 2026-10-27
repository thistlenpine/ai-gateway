"""Audit event model and sink. OWNER: observability.

Audit is a distinct stream from app logs: separate OTel logger scope `ai_gateway.audit`,
`stream="audit"` field, optional separate OTLP endpoint, never sampled, never contains tokens.
"""

from __future__ import annotations

import enum
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import IO, Any, Protocol

from opentelemetry.sdk._logs import LoggerProvider

from ai_gateway.config import AuditConfig
from ai_gateway.models import LimitDecision, Phase, PolicyDecision, RequestContext, Resource
from ai_gateway.observability import schema
from ai_gateway.observability.logging import (
    AUDIT_LOGGER_NAME,
    JsonFormatter,
    RedactingOtelHandler,
    correlation_fields,
    mark_handler,
    remove_marked_handlers,
)
from ai_gateway.observability.redact import redact_mapping, strip_query


class AuditType(enum.StrEnum):
    AUTHN_FAILURE = "authn.failure"
    AUTHZ_DECISION = "authz.decision"  # emitted for BOTH allow and deny
    RATELIMIT_DECISION = "ratelimit.decision"  # emitted for delay/reject/block (not plain allow)
    CLIENT_BLOCKED = "ratelimit.client_blocked"  # phase-3 cooldown started
    CLIENT_UNBLOCKED = "ratelimit.client_unblocked"


@dataclass(slots=True)
class AuditEvent:
    """Flat, schema-stable record. Field names are the log attribute names."""

    type: AuditType
    request_id: str
    trace_id: str = ""
    timestamp: float = field(default_factory=time.time)
    client_id: str | None = None
    subject: str | None = None
    issuer: str | None = None
    token_id: str | None = None
    decision: str | None = None  # allow | deny | delay | reject | block
    reason: str | None = None
    rule_id: str | None = None
    upstream: str | None = None
    resource_kind: str | None = None
    method: str | None = None
    path: str | None = None
    mcp_method: str | None = None
    tool: str | None = None
    model: str | None = None
    scopes_matched: list[str] = field(default_factory=list)
    scopes_missing_hint: list[str] = field(default_factory=list)
    client_ip: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict, empty values dropped. Credential keys in `extra` are masked and every
        string is scrubbed (paths lose their query string)."""
        raw = asdict(self)
        raw["type"] = self.type.value
        if isinstance(raw.get("path"), str):
            raw["path"] = strip_query(raw["path"])
        out = redact_mapping(raw)
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}


class AuditSink(Protocol):
    def emit(self, event: AuditEvent) -> None: ...

    def shutdown(self) -> None: ...


class OtelAuditSink:
    """Audit stream: stdlib logger `ai_gateway.audit` (no propagation to root) with an optional
    stdout JSON handler (`stream="audit"`) and an OTel handler on a LoggerProvider, so records
    carry instrumentation scope `ai_gateway.audit`. Never raises; never sampled."""

    def __init__(
        self,
        cfg: AuditConfig,
        *,
        logger_provider: LoggerProvider | None = None,
        stream: IO[str] | None = None,
    ) -> None:
        self._enabled = cfg.enabled
        self._log = logging.getLogger(AUDIT_LOGGER_NAME)
        remove_marked_handlers(self._log)
        self._log.setLevel(logging.INFO)
        self._log.propagate = False
        if not cfg.enabled:
            return
        if cfg.stdout:
            sh = logging.StreamHandler(stream if stream is not None else sys.stdout)
            sh.setFormatter(JsonFormatter(stream="audit"))
            self._log.addHandler(mark_handler(sh))
        if logger_provider is not None:
            self._log.addHandler(
                mark_handler(RedactingOtelHandler(logging.INFO, logger_provider, stream="audit"))
            )

    def emit(self, event: AuditEvent) -> None:
        if not self._enabled:
            return
        try:
            payload = event.to_dict()
            payload.pop("timestamp", None)  # carried by the record itself
            ids = correlation_fields()
            for k in (schema.L_REQUEST_ID, schema.L_TRACE_ID):
                if k not in payload and k in ids:
                    payload[k] = ids[k]
            record = self._log.makeRecord(
                self._log.name, logging.INFO, "(audit)", 0, event.type.value, (), None,
                extra={"audit": payload, schema.L_EVENT: event.type.value},
            )  # fmt: skip
            record.created = event.timestamp
            record.msecs = (event.timestamp % 1) * 1000
            self._log.handle(record)
        except Exception:  # telemetry must never break a request
            logging.getLogger(__name__).debug("audit emit failed", exc_info=True)

    def shutdown(self) -> None:
        remove_marked_handlers(self._log)


def _upstream_of(ctx: RequestContext, resource: Resource | None) -> str | None:
    if resource is not None:
        return resource.upstream
    return ctx.route.upstream if ctx.route is not None else None


def _base(event_type: AuditType, ctx: RequestContext, resource: Resource | None) -> AuditEvent:
    p = ctx.principal
    return AuditEvent(
        type=event_type,
        request_id=ctx.request_id,
        trace_id=ctx.trace_id,
        client_id=p.client_id if p is not None else None,
        subject=p.subject if p is not None else None,
        issuer=p.issuer if p is not None else None,
        token_id=p.token_id if p is not None else None,
        upstream=_upstream_of(ctx, resource),
        resource_kind=resource.kind.value if resource is not None else None,
        method=resource.method if resource is not None else ctx.method,
        path=strip_query(resource.path if resource is not None else ctx.path),
        mcp_method=resource.mcp_method if resource is not None else None,
        tool=resource.tool if resource is not None else None,
        model=resource.model if resource is not None else None,
        client_ip=ctx.client_ip or None,
    )


def authz_event(ctx: RequestContext, decision: PolicyDecision) -> AuditEvent:
    """Build the authz.decision event from ctx + decision."""
    ev = _base(AuditType.AUTHZ_DECISION, ctx, decision.resource)
    ev.decision = "allow" if decision.allowed else "deny"
    ev.reason = decision.reason.value
    ev.rule_id = decision.rule_id
    ev.scopes_matched = sorted(decision.matched_scopes)
    ev.scopes_missing_hint = list(decision.missing_scopes_hint)
    return ev


def ratelimit_event(ctx: RequestContext, decision: LimitDecision) -> AuditEvent:
    """Build ratelimit.decision / ratelimit.client_blocked event."""
    kind = AuditType.CLIENT_BLOCKED if decision.phase is Phase.BLOCK else AuditType.RATELIMIT_DECISION
    ev = _base(kind, ctx, ctx.resources[0] if ctx.resources else None)
    ev.client_id = ev.client_id or decision.client_id
    ev.decision = decision.phase.value
    ev.reason = decision.reason or None
    ev.extra = {
        "scope": decision.scope.value,
        "usage_ratio": round(decision.usage_ratio, 4),
        "limit": decision.limit,
        "remaining": decision.remaining,
        "retry_after_s": round(decision.retry_after_s, 3),
        "delay_s": round(decision.delay_s, 3),
    }
    return ev
