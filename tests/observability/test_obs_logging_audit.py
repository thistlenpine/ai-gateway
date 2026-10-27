from __future__ import annotations

import json
import logging
from typing import Any

from obs_rig import API_KEY, BEARER, Rig
from opentelemetry.trace import Span

from ai_gateway.context import reset_ctx, set_ctx
from ai_gateway.models import (
    DecisionReason,
    LimitDecision,
    LimitScope,
    Phase,
    PolicyDecision,
    Principal,
    RequestContext,
    Resource,
    UpstreamKind,
)
from ai_gateway.observability import AuditEvent, AuditType
from ai_gateway.observability.audit import authz_event, ratelimit_event
from ai_gateway.observability.logging import AUDIT_LOGGER_NAME


def _lines(rig: Rig) -> list[dict[str, Any]]:
    return [json.loads(x) for x in rig.stdout.getvalue().splitlines() if x.strip()]


def _ctx() -> RequestContext:
    p = Principal("svc-a", "sub-1", "https://idp", frozenset({"x"}), token_id="jti-1")
    return RequestContext(
        "req-12345678", "POST", "/mcp/tools?access_token=" + BEARER, "10.0.0.1", "a" * 32, principal=p
    )


def test_json_log_has_correlation_ids_and_redacts(rig: Rig) -> None:
    ctx = _ctx()
    token = set_ctx(ctx)
    try:
        with rig.tel.span("gateway.authn") as span:
            sc = span.get_span_context()
            logging.getLogger("ai_gateway.test").info(
                "validating Authorization: Bearer %s",
                BEARER,
                extra={
                    "event": "authn.test",
                    "headers": {"Authorization": f"Bearer {BEARER}"},
                    "x-api-key": API_KEY,
                    "note": f"key={API_KEY}",
                    "status": 200,
                },
            )
    finally:
        reset_ctx(token)
    (line,) = _lines(rig)
    assert line["trace_id"] == f"{sc.trace_id:032x}"
    assert line["span_id"] == f"{sc.span_id:016x}"
    assert line["request_id"] == "req-12345678"
    assert line["client_id"] == "anonymous"
    assert line["stream"] == "app" and line["event"] == "authn.test" and line["level"] == "INFO"
    assert line["status"] == 200
    assert line["headers"]["Authorization"] == "[REDACTED]"
    assert line["x-api-key"] == "[REDACTED]"
    assert BEARER not in rig.everything() and API_KEY not in rig.everything()


def test_exception_text_is_scrubbed(rig: Rig) -> None:
    try:
        raise ValueError(f"bad token {BEARER} and api_key={API_KEY}")
    except ValueError:
        logging.getLogger("ai_gateway.test").error("boom", exc_info=True)
    (line,) = _lines(rig)
    assert "ValueError" in line["exception"]
    assert BEARER not in rig.everything() and API_KEY not in rig.everything()


def test_logs_exported_via_otel_with_trace_and_request_id(rig: Rig) -> None:
    token = set_ctx(_ctx())
    try:
        with rig.tel.span("s") as span:
            logging.getLogger("ai_gateway.test").warning(
                "hello", extra={"event": "x.y", "authorization": "Bearer abc"}
            )
    finally:
        reset_ctx(token)
    recs = [r for r in rig.log_records() if r[2].body == "hello"]
    assert len(recs) == 1
    _, attrs, rec = recs[0]
    assert attrs["event"] == "x.y" and attrs["authorization"] == "[REDACTED]"
    assert attrs["request_id"] == "req-12345678"
    assert rec.trace_id == span.get_span_context().trace_id


def test_audit_event_stream(rig: Rig) -> None:
    ev = AuditEvent(
        type=AuditType.AUTHN_FAILURE,
        request_id="req-12345678",
        trace_id="b" * 32,
        decision="deny",
        reason="expired",
        path=f"/p?api_key={API_KEY}",
        extra={"authorization": f"Bearer {BEARER}", "detail": f"token={BEARER}", "attempt": 2},
    )
    rig.tel.audit(ev)
    audit_logs = rig.log_records(AUDIT_LOGGER_NAME)
    assert len(audit_logs) == 1
    _, attrs, rec = audit_logs[0]
    assert rec.body == "authn.failure"
    assert attrs["stream"] == "audit" and attrs["event"] == "authn.failure"
    assert attrs["type"] == "authn.failure" and attrs["decision"] == "deny"
    assert attrs["request_id"] == "req-12345678" and attrs["trace_id"] == "b" * 32
    assert attrs["path"] == "/p"
    (line,) = _lines(rig)  # exactly one stdout line: audit logger does not propagate to root
    assert line["stream"] == "audit" and line["event"] == "authn.failure"
    assert line["extra"]["authorization"] == "[REDACTED]" and line["extra"]["attempt"] == 2
    assert "api_key" not in line["path"]
    assert BEARER not in rig.everything() and API_KEY not in rig.everything()


def test_audit_fills_trace_and_request_ids_from_context(rig: Rig) -> None:
    token = set_ctx(_ctx())
    try:
        with rig.tel.span("s") as span:
            rig.tel.audit(AuditEvent(type=AuditType.CLIENT_BLOCKED, request_id=""))
    finally:
        reset_ctx(token)
    (line,) = _lines(rig)
    assert line["trace_id"] == f"{span.get_span_context().trace_id:032x}"
    assert line["request_id"] == "req-12345678"


def test_audit_never_in_app_stream_and_app_logs_not_audit_scope(rig: Rig) -> None:
    logging.getLogger("ai_gateway.pipeline").info("plain")
    assert rig.log_records(AUDIT_LOGGER_NAME) == []


def test_authz_event_from_ctx_and_decision() -> None:
    res = Resource(
        UpstreamKind.MCP, "tools", "POST", "/mcp", mcp_method="tools/call", tool="search", model=None
    )
    dec = PolicyDecision(
        False,
        DecisionReason.INSUFFICIENT_SCOPE,
        res,
        "r1",
        frozenset({"b", "a"}),
        missing_scopes_hint=("tools:call",),
    )
    ev = authz_event(_ctx(), dec)
    assert ev.type is AuditType.AUTHZ_DECISION
    assert (ev.decision, ev.reason, ev.rule_id) == ("deny", "insufficient_scope", "r1")
    assert ev.client_id == "svc-a" and ev.subject == "sub-1" and ev.token_id == "jti-1"
    assert ev.upstream == "tools" and ev.tool == "search" and ev.mcp_method == "tools/call"
    assert ev.scopes_matched == ["a", "b"] and ev.scopes_missing_hint == ["tools:call"]
    assert ev.path == "/mcp" and ev.client_ip == "10.0.0.1" and ev.request_id == "req-12345678"
    allow = authz_event(_ctx(), PolicyDecision(True, DecisionReason.ALLOWED, res))
    assert allow.decision == "allow"
    d = ev.to_dict()
    assert d["type"] == "authz.decision" and "model" not in d
    assert BEARER not in json.dumps(authz_event(_ctx(), dec).to_dict())  # ctx.path query stripped


def test_ratelimit_event_types() -> None:
    def mk(ph: Phase) -> LimitDecision:
        return LimitDecision(ph, LimitScope.REQUESTS, "svc-a", 10, 0, 1.0, usage_ratio=1.6, retry_after_s=2.0)

    assert ratelimit_event(_ctx(), mk(Phase.REJECT)).type is AuditType.RATELIMIT_DECISION
    blocked = ratelimit_event(_ctx(), mk(Phase.BLOCK))
    assert blocked.type is AuditType.CLIENT_BLOCKED and blocked.decision == "block"
    assert blocked.extra["scope"] == "requests" and blocked.extra["usage_ratio"] == 1.6


def test_span_events_unaffected_type_check(rig: Rig) -> None:
    with rig.tel.span("x") as s:
        assert isinstance(s, Span)
