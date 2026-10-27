"""OTel metric instruments behind the Telemetry.record_* methods. OWNER: observability."""

from __future__ import annotations

from typing import Final

from opentelemetry.metrics import Meter
from opentelemetry.util.types import AttributeValue

from ai_gateway.models import LimitDecision, Phase, PolicyDecision, RequestContext, UsageRecord
from ai_gateway.observability import schema

DURATION_BUCKETS: Final = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

_HTTP_METHODS: Final = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "CONNECT", "TRACE"}
)
_MAX_LABEL_LEN: Final = 64


class _Bounded:
    """Cardinality guard for labels that can carry client-influenced values (issuer from an
    unverified token, model name from a request body, error type text...). After `limit`
    distinct values every new value collapses to "other"; long values are truncated."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._seen: set[str] = set()

    def __call__(self, value: str | None, default: str = "unknown") -> str:
        if not value:
            return default
        value = value[:_MAX_LABEL_LEN]
        if value in self._seen:
            return value
        if len(self._seen) >= self._limit:
            return "other"
        self._seen.add(value)
        return value


Attrs = dict[str, AttributeValue]


class Metrics:
    """Creates all instruments named in schema.py once; one method per Telemetry.record_*.

    Histogram bucket boundaries: request/upstream duration `DURATION_BUCKETS`, set as explicit
    bucket advisory on the instrument so they apply with any MeterProvider.

    Labels are low cardinality by construction: route/upstream ids come from config, client is
    `ctx.client_label` (registered id | unregistered | anonymous), paths are never labels, and
    client-influenced values (issuer, model, error type) pass through a `_Bounded` guard.
    """

    def __init__(self, meter: Meter) -> None:
        b = list(DURATION_BUCKETS)
        self._requests = meter.create_counter(schema.M_REQUESTS, unit="{request}")
        self._duration = meter.create_histogram(
            schema.M_REQUEST_DURATION, unit="s", explicit_bucket_boundaries_advisory=b
        )
        self._active = meter.create_up_down_counter(schema.M_ACTIVE, unit="{request}")
        self._rl = meter.create_counter(schema.M_RL_DECISIONS, unit="{decision}")
        self._rl_delay = meter.create_histogram(
            schema.M_RL_DELAY, unit="s", explicit_bucket_boundaries_advisory=b
        )
        self._auth_fail = meter.create_counter(schema.M_AUTH_FAILURES, unit="{failure}")
        self._authz = meter.create_counter(schema.M_AUTHZ_DECISIONS, unit="{decision}")
        self._up_duration = meter.create_histogram(
            schema.M_UPSTREAM_DURATION, unit="s", explicit_bucket_boundaries_advisory=b
        )
        self._up_errors = meter.create_counter(schema.M_UPSTREAM_ERRORS, unit="{error}")
        self._tokens = meter.create_counter(schema.M_LLM_TOKENS, unit="{token}")
        self._jwks = meter.create_counter(schema.M_JWKS_REFRESH, unit="{refresh}")
        self._issuer = _Bounded(32)
        self._model = _Bounded(64)
        self._error = _Bounded(32)
        self._upstream = _Bounded(128)
        self._client = _Bounded(256)
        self._other = _Bounded(64)

    # -- helpers
    def _client_label(self, ctx: RequestContext | None, fallback: str | None = None) -> str:
        if ctx is not None:
            return self._client(ctx.client_label, "anonymous")
        return self._client(fallback, "anonymous")

    def _route(self, ctx: RequestContext) -> tuple[str, str]:
        if ctx.route is None:
            return "none", "none"
        return self._upstream(ctx.route.upstream, "none"), str(ctx.route.kind.value)

    # -- instruments
    def add_active(self, delta: int, upstream: str | None) -> None:
        attrs: Attrs = {schema.A_UPSTREAM: self._upstream(upstream)} if upstream else {}
        self._active.add(delta, attrs)

    def request(self, ctx: RequestContext) -> None:
        upstream, kind = self._route(ctx)
        method = ctx.method.upper()
        attrs: Attrs = {
            schema.A_HTTP_METHOD: method if method in _HTTP_METHODS else "_OTHER",
            schema.A_HTTP_STATUS: ctx.status_code or 0,
            schema.A_UPSTREAM: upstream,
            schema.A_UPSTREAM_KIND: kind,
            schema.A_CLIENT_ID: self._client_label(ctx),
        }
        self._requests.add(1, attrs)
        self._duration.record(max(ctx.elapsed_s, 0.0), attrs)

    def auth_failure(self, reason: str, issuer: str | None) -> None:
        self._auth_fail.add(
            1,
            {
                schema.A_AUTH_REASON: self._other(reason),
                schema.A_ISSUER: self._issuer(issuer),
            },
        )

    def authz(self, decision: PolicyDecision, ctx: RequestContext) -> None:
        if decision.resource is not None:
            upstream = self._upstream(decision.resource.upstream, "none")
        else:
            upstream = self._route(ctx)[0]
        self._authz.add(
            1,
            {
                schema.A_AUTHZ_DECISION: "allow" if decision.allowed else "deny",
                schema.A_AUTHZ_REASON: decision.reason.value,
                schema.A_UPSTREAM: upstream,
            },
        )

    def rate_limit(self, decision: LimitDecision, ctx: RequestContext | None) -> None:
        client = self._client_label(ctx, decision.client_id)
        self._rl.add(
            1,
            {
                schema.A_CLIENT_ID: client,
                schema.A_RL_PHASE: decision.phase.value,
                schema.A_RL_SCOPE: decision.scope.value,
            },
        )
        if decision.phase is Phase.DELAY:
            self._rl_delay.record(max(decision.delay_s, 0.0), {schema.A_CLIENT_ID: client})

    def upstream(
        self, upstream: str, kind: str, duration_s: float, status: int | None, error_type: str | None
    ) -> None:
        up = self._upstream(upstream)
        kind = self._other(kind)
        attrs: Attrs = {schema.A_UPSTREAM: up, schema.A_UPSTREAM_KIND: kind}
        if status is not None:
            attrs[schema.A_HTTP_STATUS] = status
        self._up_duration.record(max(duration_s, 0.0), attrs)
        if error_type or (status is not None and status >= 500):
            self._up_errors.add(
                1,
                {
                    schema.A_UPSTREAM: up,
                    schema.A_UPSTREAM_KIND: kind,
                    schema.A_ERROR_TYPE: self._error(error_type or str(status)),
                },
            )

    def llm_tokens(self, usage: UsageRecord, upstream: str, client_label: str) -> None:
        base: Attrs = {
            schema.A_LLM_MODEL: self._model(usage.model),
            schema.A_LLM_SYSTEM: self._other(usage.provider),
            schema.A_CLIENT_ID: self._client(client_label, "anonymous"),
            schema.A_UPSTREAM: self._upstream(upstream),
        }
        for token_type, n in (("input", usage.input_tokens), ("output", usage.output_tokens)):
            if n > 0:
                self._tokens.add(n, {**base, schema.A_LLM_TOKEN_TYPE: token_type})

    def jwks_refresh(self, issuer: str, outcome: str) -> None:
        self._jwks.add(
            1, {schema.A_ISSUER: self._issuer(issuer), schema.A_JWKS_OUTCOME: self._other(outcome)}
        )
