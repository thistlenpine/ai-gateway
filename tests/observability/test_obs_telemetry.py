from __future__ import annotations

import pytest
from obs_rig import API_KEY, BEARER, Rig, attrs_of
from opentelemetry.sdk.metrics.export import HistogramDataPoint, NumberDataPoint
from opentelemetry.trace import SpanKind, StatusCode

from ai_gateway.config import TelemetryConfig
from ai_gateway.context import reset_ctx, set_ctx
from ai_gateway.models import (
    DecisionReason,
    LimitDecision,
    LimitScope,
    Phase,
    PolicyDecision,
    RequestContext,
    Resource,
    RouteMatch,
    UpstreamKind,
    UsageRecord,
)
from ai_gateway.observability import NullTelemetry, Telemetry, schema, setup_telemetry
from ai_gateway.observability.metrics import DURATION_BUCKETS
from ai_gateway.observability.otel import OtelTelemetry


def _ctx(status: int = 200) -> RequestContext:
    c = RequestContext("req-12345678", "GET", "/billing/invoices/42?x=1", "1.1.1.1", "a" * 32)
    c.route = RouteMatch("billing", UpstreamKind.HTTP, "/billing", "/invoices/42")
    c.client_label = "svc-a"
    c.status_code = status
    return c


def test_span_parent_child_and_attributes(rig: Rig) -> None:
    token = set_ctx(_ctx())
    try:
        with (
            rig.tel.span("gateway.proxy", {schema.A_UPSTREAM: "billing"}),
            rig.tel.span("gateway.authz"),
        ):
            pass
    finally:
        reset_ctx(token)
    spans = {s.name: s for s in rig.spans.get_finished_spans()}
    proxy, authz = spans["gateway.proxy"], spans["gateway.authz"]
    assert authz.parent is not None and authz.parent.span_id == proxy.context.span_id
    assert authz.context.trace_id == proxy.context.trace_id
    assert proxy.parent is None and proxy.kind is SpanKind.INTERNAL
    assert attrs_of(proxy)[schema.A_UPSTREAM] == "billing"
    assert attrs_of(proxy)[schema.A_REQUEST_ID] == "req-12345678"
    assert attrs_of(proxy)[schema.A_CLIENT_ID] == "svc-a"
    assert proxy.resource.attributes["service.name"] == "ai-gateway"
    assert proxy.resource.attributes["deployment.environment"] == "dev"


def test_span_error_status_and_redaction(rig: Rig) -> None:
    with (
        pytest.raises(RuntimeError),
        rig.tel.span("gateway.authn", {"authorization": f"Bearer {BEARER}"}) as s,
    ):
        s.set_attribute("x-api-key", API_KEY)
        s.set_attribute("note", f"api_key={API_KEY}")
        raise RuntimeError(f"jwt {BEARER} failed")
    (span,) = rig.spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert attrs_of(span)["authorization"] == "[REDACTED]" and attrs_of(span)["x-api-key"] == "[REDACTED]"
    exc = next(e for e in span.events if e.name == "exception")
    assert (exc.attributes or {})["exception.type"] == "RuntimeError"
    assert BEARER not in rig.everything() and API_KEY not in rig.everything()


def test_request_metrics_names_and_attrs(rig: Rig) -> None:
    rig.tel.record_request(_ctx(200))
    rig.tel.add_active(1, upstream="billing")
    assert {schema.M_REQUESTS, schema.M_REQUEST_DURATION, schema.M_ACTIVE} <= rig.metric_names()
    (attrs, pt) = rig.metric(schema.M_REQUESTS)[0]
    assert attrs == {
        schema.A_HTTP_METHOD: "GET",
        schema.A_HTTP_STATUS: 200,
        schema.A_UPSTREAM: "billing",
        schema.A_UPSTREAM_KIND: "http",
        schema.A_CLIENT_ID: "svc-a",
    }
    assert isinstance(pt, NumberDataPoint) and pt.value == 1
    (_, hist) = rig.metric(schema.M_REQUEST_DURATION)[0]
    assert isinstance(hist, HistogramDataPoint) and tuple(hist.explicit_bounds) == DURATION_BUCKETS
    # path / ids never labels
    for name in (schema.M_REQUESTS, schema.M_REQUEST_DURATION):
        for attrs, _ in rig.metric(name):
            assert "/invoices/42" not in str(attrs.values())
    (aa, ap) = rig.metric(schema.M_ACTIVE)[0]
    assert aa == {schema.A_UPSTREAM: "billing"} and ap.value == 1


def test_auth_authz_ratelimit_metrics(rig: Rig) -> None:
    ctx = _ctx()
    rig.tel.record_auth_failure("expired", issuer="https://idp")
    res = Resource(UpstreamKind.HTTP, "billing", "GET", "/invoices/42")
    rig.tel.record_authz(PolicyDecision(False, DecisionReason.NO_MATCHING_RULE, res), ctx)
    rig.tel.record_authz(PolicyDecision(True, DecisionReason.ALLOWED, res), ctx)
    rig.tel.record_rate_limit(
        LimitDecision(Phase.DELAY, LimitScope.REQUESTS, "raw-client-id", 10, 1, 1, delay_s=0.3), ctx
    )
    rig.tel.record_rate_limit(LimitDecision(Phase.REJECT, LimitScope.TOKENS, "raw-client-id", 10, 0, 1))
    assert rig.metric(schema.M_AUTH_FAILURES)[0][0] == {
        schema.A_AUTH_REASON: "expired",
        schema.A_ISSUER: "https://idp",
    }
    authz = {tuple(sorted(a.items())) for a, _ in rig.metric(schema.M_AUTHZ_DECISIONS)}
    assert (
        (schema.A_AUTHZ_DECISION, "deny"),
        (schema.A_AUTHZ_REASON, "no_matching_rule"),
        (schema.A_UPSTREAM, "billing"),
    ) in authz
    rl = {
        (a[schema.A_CLIENT_ID], a[schema.A_RL_PHASE], a[schema.A_RL_SCOPE])
        for a, _ in rig.metric(schema.M_RL_DECISIONS)
    }
    assert rl == {("svc-a", "delay", "requests"), ("raw-client-id", "reject", "tokens")}
    (da, dp) = rig.metric(schema.M_RL_DELAY)[0]
    assert (
        da == {schema.A_CLIENT_ID: "svc-a"}
        and isinstance(dp, HistogramDataPoint)
        and dp.sum == pytest.approx(0.3)
    )


def test_upstream_and_llm_and_jwks_metrics(rig: Rig) -> None:
    rig.tel.record_upstream(upstream="billing", kind="http", duration_s=0.2, status=200)
    rig.tel.record_upstream(upstream="billing", kind="http", duration_s=0.2, status=503)
    rig.tel.record_upstream(upstream="llm", kind="llm", duration_s=1.0, status=None, error_type="timeout")
    dur = [a for a, _ in rig.metric(schema.M_UPSTREAM_DURATION)]
    assert {schema.A_HTTP_STATUS: 200, schema.A_UPSTREAM: "billing", schema.A_UPSTREAM_KIND: "http"} in dur
    errs = {a[schema.A_ERROR_TYPE] for a, _ in rig.metric(schema.M_UPSTREAM_ERRORS)}
    assert errs == {"503", "timeout"}
    rig.tel.record_llm_tokens(
        UsageRecord(100, 20, "claude-x", "anthropic"), upstream="llm", client_label="svc-a"
    )
    toks = {a[schema.A_LLM_TOKEN_TYPE]: (a, p.value) for a, p in rig.metric(schema.M_LLM_TOKENS)}
    assert toks["input"][1] == 100 and toks["output"][1] == 20
    assert toks["input"][0] == {
        schema.A_LLM_TOKEN_TYPE: "input",
        schema.A_LLM_MODEL: "claude-x",
        schema.A_LLM_SYSTEM: "anthropic",
        schema.A_CLIENT_ID: "svc-a",
        schema.A_UPSTREAM: "llm",
    }
    rig.tel.record_jwks_refresh(issuer="https://idp", outcome="refreshed")
    assert rig.metric(schema.M_JWKS_REFRESH)[0][0] == {
        schema.A_ISSUER: "https://idp",
        schema.A_JWKS_OUTCOME: "refreshed",
    }


def test_client_influenced_labels_are_bounded(rig: Rig) -> None:
    for i in range(300):
        rig.tel.record_auth_failure("invalid_token", issuer=f"https://evil-{i}.example")
        rig.tel.record_llm_tokens(UsageRecord(1, 1, f"model-{i}", "openai"), upstream="llm", client_label="c")
    issuers = {a[schema.A_ISSUER] for a, _ in rig.metric(schema.M_AUTH_FAILURES)}
    models = {a[schema.A_LLM_MODEL] for a, _ in rig.metric(schema.M_LLM_TOKENS)}
    assert len(issuers) <= 33 and "other" in issuers
    assert len(models) <= 65 and "other" in models
    rig.tel.record_request(_ctx())
    long = _ctx()
    long.method = "BREW"
    rig.tel.record_request(long)
    assert "_OTHER" in {a[schema.A_HTTP_METHOD] for a, _ in rig.metric(schema.M_REQUESTS)}


def test_all_schema_metrics_are_created(rig: Rig) -> None:
    ctx = _ctx()
    res = Resource(UpstreamKind.HTTP, "billing", "GET", "/x")
    rig.tel.record_request(ctx)
    rig.tel.add_active(1)
    rig.tel.record_auth_failure("x")
    rig.tel.record_authz(PolicyDecision(True, DecisionReason.ALLOWED, res), ctx)
    rig.tel.record_rate_limit(LimitDecision(Phase.DELAY, LimitScope.REQUESTS, "c", 1, 0, 0, delay_s=0.1), ctx)
    rig.tel.record_upstream(upstream="u", kind="http", duration_s=0.1, status=500)
    rig.tel.record_llm_tokens(UsageRecord(1, 1), upstream="u", client_label="c")
    rig.tel.record_jwks_refresh(issuer="i", outcome="hit")
    expected = {v for k, v in vars(schema).items() if k.startswith("M_")}
    assert expected <= rig.metric_names()


def test_setup_telemetry_disabled_returns_null() -> None:
    t = setup_telemetry(TelemetryConfig(enabled=False, log_stdout_json=False))
    assert isinstance(t, NullTelemetry) and isinstance(t, Telemetry)


async def test_setup_telemetry_without_endpoint_is_local_noop_export() -> None:
    cfg = TelemetryConfig.model_validate({"otlp": {"endpoint": ""}, "log_stdout_json": False})
    t = setup_telemetry(cfg)
    assert isinstance(t, OtelTelemetry) and isinstance(t, Telemetry)
    with t.span("x") as s:
        assert s.get_span_context().is_valid  # spans still exist for correlation
    t.record_request(_ctx())
    await t.shutdown()
    await t.shutdown()  # idempotent


def test_null_telemetry_still_collects_audit() -> None:
    from ai_gateway.observability import AuditEvent, AuditType

    n = NullTelemetry()
    n.audit(AuditEvent(AuditType.AUTHN_FAILURE, "r"))
    with n.span("x"):
        pass
    assert len(n.audit_events) == 1
