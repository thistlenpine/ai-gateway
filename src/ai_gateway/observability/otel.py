"""OpenTelemetry SDK bootstrap + concrete Telemetry. OWNER: observability. OTLP only."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import IO, Any, Final

import httpx
from fastapi import FastAPI
from opentelemetry import propagate
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagators.textmap import Setter, default_setter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, SpanContext, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import Attributes, AttributeValue
from starlette.types import Scope

from ai_gateway.config import OtlpConfig, TelemetryConfig
from ai_gateway.context import get_ctx
from ai_gateway.models import LimitDecision, PolicyDecision, RequestContext, UsageRecord
from ai_gateway.observability import schema
from ai_gateway.observability.audit import AuditEvent, OtelAuditSink
from ai_gateway.observability.logging import configure_logging, shutdown_logging
from ai_gateway.observability.metrics import Metrics
from ai_gateway.observability.redact import is_sensitive_key, redact_value, scrub_query, scrub_text, scrub_url
from ai_gateway.observability.telemetry import AttrValue, NullTelemetry, Telemetry

_log = logging.getLogger(__name__)

_SCOPE_NAME: Final = "ai_gateway"
_EXCLUDED_URLS: Final = "/healthz,/readyz"
_URL_KEYS: Final = ("http.url", "url.full", "http.target", "url.original")
_QUERY_KEYS: Final = ("url.query",)


# --------------------------------------------------------------------------- safe span


class _SafeSpan(Span):
    """Span wrapper that redacts attributes, events and exception text before they reach the SDK."""

    def __init__(self, inner: Span) -> None:
        self._inner = inner

    def get_span_context(self) -> SpanContext:
        return self._inner.get_span_context()

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for k, v in attributes.items():
            self.set_attribute(k, v)

    def set_attribute(self, key: str, value: Any) -> None:
        self._inner.set_attribute(key, _safe_attr(key, value))

    def add_event(self, name: str, attributes: Attributes = None, timestamp: int | None = None) -> None:
        self._inner.add_event(name, _safe_attrs(attributes), timestamp)

    def update_name(self, name: str) -> None:
        self._inner.update_name(scrub_text(name))

    def is_recording(self) -> bool:
        return self._inner.is_recording()

    def set_status(self, status: Status | StatusCode, description: str | None = None) -> None:
        if isinstance(status, Status):
            description = status.description
            status = status.status_code
        self._inner.set_status(status, scrub_text(description) if description else None)

    def record_exception(
        self,
        exception: BaseException,
        attributes: Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        # Exception messages can embed credentials: record type + scrubbed message only.
        attrs: dict[str, AttributeValue] = {
            "exception.type": type(exception).__qualname__,
            "exception.message": scrub_text(str(exception)),
            "exception.escaped": escaped,
            **_safe_attrs(attributes),
        }
        self._inner.add_event("exception", attrs, timestamp)

    def end(self, end_time: int | None = None) -> None:
        self._inner.end(end_time)


def _safe_attr(key: str, value: Any) -> Any:
    out = redact_value(key, value)
    if isinstance(out, dict):
        return str(out)
    return out


def _safe_attrs(attributes: Attributes) -> dict[str, AttributeValue]:
    return {k: _safe_attr(k, v) for k, v in (attributes or {}).items()}


# --------------------------------------------------------------------------- exporters


def _headers(cfg: OtlpConfig) -> dict[str, str] | None:
    """Parse "k=v,k2=v2" from the env var named by cfg.headers_env (never inline in config)."""
    if not cfg.headers_env:
        return None
    raw = os.environ.get(cfg.headers_env, "")
    pairs = (p.partition("=") for p in raw.split(",") if "=" in p)
    out = {k.strip(): v.strip() for k, _, v in pairs if k.strip()}
    return out or None


def _http_endpoint(base: str, signal: str) -> str:
    return f"{base.rstrip('/')}/v1/{signal}"


def _span_exporter(cfg: OtlpConfig) -> SpanExporter:
    if cfg.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as G

        return G(
            endpoint=cfg.endpoint, insecure=cfg.insecure, headers=_headers(cfg), timeout=int(cfg.timeout_s)
        )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as H

    return H(
        endpoint=_http_endpoint(cfg.endpoint, "traces"), headers=_headers(cfg), timeout=int(cfg.timeout_s)
    )


def _metric_exporter(cfg: OtlpConfig) -> Any:
    if cfg.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as G

        return G(
            endpoint=cfg.endpoint, insecure=cfg.insecure, headers=_headers(cfg), timeout=int(cfg.timeout_s)
        )
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as H

    return H(
        endpoint=_http_endpoint(cfg.endpoint, "metrics"), headers=_headers(cfg), timeout=int(cfg.timeout_s)
    )


def _log_exporter(cfg: OtlpConfig, endpoint: str) -> LogRecordExporter:
    if cfg.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as G

        return G(endpoint=endpoint, insecure=cfg.insecure, headers=_headers(cfg), timeout=int(cfg.timeout_s))
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as H

    return H(endpoint=_http_endpoint(endpoint, "logs"), headers=_headers(cfg), timeout=int(cfg.timeout_s))


# --------------------------------------------------------------------------- telemetry

_ACTIVE: OtelTelemetry | None = None


class _DropTracestate(Setter[Any]):
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def set(self, carrier: Any, key: str, value: str) -> None:
        if key.lower() != "tracestate":
            self._inner.set(carrier, key, value)


class _TraceparentOnlyPropagator(TraceContextTextMapPropagator):
    """W3C traceparent only: inbound `tracestate` (vendor data chosen by the caller) is never
    re-injected on upstream requests, and baggage is not propagated at all."""

    def inject(self, carrier: Any, context: Any = None, setter: Any = default_setter) -> None:
        super().inject(carrier, context, _DropTracestate(setter))


class OtelTelemetry:
    """Implements `Telemetry` on the OTel SDK (TracerProvider, MeterProvider, LoggerProvider,
    OTLP exporters per cfg.otlp, Resource{service.name, service.version,
    deployment.environment}, ParentBased(TraceIdRatioBased(cfg.trace_sampler_ratio)), W3C
    tracecontext+baggage propagator). Delegates metrics to metrics.Metrics, audit to
    audit sink.

    With an empty `cfg.otlp.endpoint` nothing is exported (spans still exist for correlation,
    logs/audit still go to stdout). The keyword-only exporter/reader arguments replace the OTLP
    exporters (used by tests with in-memory implementations, wired with synchronous processors).
    Providers are NOT installed as OTel globals; `instrument_app` picks them up from here.
    """

    def __init__(
        self,
        cfg: TelemetryConfig,
        *,
        span_exporter: SpanExporter | None = None,
        metric_reader: MetricReader | None = None,
        log_exporter: LogRecordExporter | None = None,
        stdout: IO[str] | None = None,
    ) -> None:
        global _ACTIVE
        self._cfg = cfg
        resource = Resource.create(
            {
                "service.name": cfg.service_name,
                "service.version": cfg.service_version,
                "deployment.environment": cfg.environment,
            }
        )
        export = bool(cfg.otlp.endpoint)

        span_processors: list[SpanProcessor] = []
        if span_exporter is not None:
            span_processors.append(SimpleSpanProcessor(span_exporter))
        elif export:
            span_processors.append(BatchSpanProcessor(_span_exporter(cfg.otlp)))
        self.tracer_provider = TracerProvider(
            sampler=ParentBased(TraceIdRatioBased(cfg.trace_sampler_ratio)),
            resource=resource,
        )
        for sp in span_processors:
            self.tracer_provider.add_span_processor(sp)
        self._tracer = self.tracer_provider.get_tracer(_SCOPE_NAME, cfg.service_version)

        readers: list[MetricReader] = []
        if metric_reader is not None:
            readers.append(metric_reader)
        elif export:
            readers.append(
                PeriodicExportingMetricReader(
                    _metric_exporter(cfg.otlp),
                    export_interval_millis=cfg.metrics_export_interval_s * 1000,
                )
            )
        self.meter_provider = MeterProvider(metric_readers=readers, resource=resource)
        self._metrics = Metrics(self.meter_provider.get_meter(_SCOPE_NAME, cfg.service_version))

        self.logger_provider = LoggerProvider(resource=resource)
        if log_exporter is not None:
            self.logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
        elif export:
            self.logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(_log_exporter(cfg.otlp, cfg.otlp.endpoint))
            )
        self._audit_provider: LoggerProvider | None = None
        audit_provider = self.logger_provider
        if cfg.audit.otlp_endpoint and log_exporter is None:
            self._audit_provider = LoggerProvider(resource=resource)
            self._audit_provider.add_log_record_processor(
                BatchLogRecordProcessor(_log_exporter(cfg.otlp, cfg.audit.otlp_endpoint))
            )
            audit_provider = self._audit_provider

        propagate.set_global_textmap(
            _TraceparentOnlyPropagator()  # no baggage/tracestate: attacker-chosen (SEC-14)
        )
        configure_logging(cfg, logger_provider=self.logger_provider, stream=stdout)
        self._audit = OtelAuditSink(cfg.audit, logger_provider=audit_provider, stream=stdout)
        self._closed = False
        _ACTIVE = self

    # -- tracing
    @contextmanager
    def _span(self, name: str, attributes: Mapping[str, AttrValue] | None) -> Iterator[Span]:
        attrs: dict[str, AttributeValue] = {}
        ctx = get_ctx()
        if ctx is not None:
            attrs[schema.A_REQUEST_ID] = ctx.request_id
            attrs[schema.A_CLIENT_ID] = ctx.client_label
        attrs.update(_safe_attrs(attributes))
        with self._tracer.start_as_current_span(
            name, attributes=attrs, record_exception=False, set_status_on_exception=False
        ) as raw:
            safe = _SafeSpan(raw)
            try:
                yield safe
            except Exception as exc:
                safe.record_exception(exc, escaped=True)
                safe.set_status(StatusCode.ERROR, type(exc).__name__)
                raise

    def span(
        self, name: str, attributes: Mapping[str, AttrValue] | None = None
    ) -> AbstractContextManager[Span]:
        return self._span(name, attributes)

    # -- metrics (telemetry must never break a request)
    def _guard(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            _log.debug("telemetry call failed", exc_info=True)

    def add_active(self, delta: int, *, upstream: str | None = None) -> None:
        self._guard(lambda: self._metrics.add_active(delta, upstream))

    def record_request(self, ctx: RequestContext) -> None:
        self._guard(lambda: self._metrics.request(ctx))

    def record_auth_failure(self, reason: str, *, issuer: str | None = None) -> None:
        self._guard(lambda: self._metrics.auth_failure(reason, issuer))

    def record_authz(self, decision: PolicyDecision, ctx: RequestContext) -> None:
        self._guard(lambda: self._metrics.authz(decision, ctx))

    def record_rate_limit(self, decision: LimitDecision, ctx: RequestContext | None = None) -> None:
        self._guard(lambda: self._metrics.rate_limit(decision, ctx))

    def record_upstream(
        self,
        *,
        upstream: str,
        kind: str,
        duration_s: float,
        status: int | None,
        error_type: str | None = None,
    ) -> None:
        self._guard(lambda: self._metrics.upstream(upstream, kind, duration_s, status, error_type))

    def record_llm_tokens(self, usage: UsageRecord, *, upstream: str, client_label: str) -> None:
        self._guard(lambda: self._metrics.llm_tokens(usage, upstream, client_label))

    def record_jwks_refresh(self, *, issuer: str, outcome: str) -> None:
        self._guard(lambda: self._metrics.jwks_refresh(issuer, outcome))

    def audit(self, event: AuditEvent) -> None:
        self._audit.emit(event)

    def _flush_and_stop(self) -> None:
        for provider in (
            self.tracer_provider,
            self.meter_provider,
            self.logger_provider,
            self._audit_provider,
        ):
            if provider is None:
                continue
            try:
                provider.shutdown()
            except Exception:
                _log.debug("telemetry shutdown failed", exc_info=True)

    async def shutdown(self) -> None:
        """Flush + stop exporters (blocking SDK calls run in a worker thread)."""
        global _ACTIVE
        if self._closed:
            return
        self._closed = True
        self._audit.shutdown()
        await asyncio.to_thread(self._flush_and_stop)
        shutdown_logging()
        if _ACTIVE is self:
            _ACTIVE = None


def setup_telemetry(cfg: TelemetryConfig) -> Telemetry:
    """Entry point used by services.build_services. cfg.enabled False -> NullTelemetry (JSON
    stdout logging is still configured). Otherwise OtelTelemetry, which also configures logging."""
    if not cfg.enabled:
        configure_logging(cfg)
        return NullTelemetry()
    return OtelTelemetry(cfg)


# --------------------------------------------------------------------------- instrumentation


def _scrub_span_urls(span: Span) -> None:
    attrs: Mapping[str, Any] = getattr(span, "attributes", None) or {}
    for key in _URL_KEYS:
        v = attrs.get(key)
        if isinstance(v, str):
            span.set_attribute(key, scrub_url(v))
    for key in _QUERY_KEYS:
        v = attrs.get(key)
        if isinstance(v, str):
            span.set_attribute(key, scrub_query(v))
    for key, v in attrs.items():
        if isinstance(v, str) and (is_sensitive_key(key) or scrub_text(v) != v):
            span.set_attribute(key, redact_value(key, v))


def _server_request_hook(span: Span, scope: Scope) -> None:
    if span is not None and span.is_recording():
        _scrub_span_urls(span)


def _client_request_hook(span: Span, request: Any) -> None:
    if span is not None and span.is_recording():
        _scrub_span_urls(span)


async def _async_client_request_hook(span: Span, request: Any) -> None:
    _client_request_hook(span, request)


def _client_response_hook(span: Span, request: Any, response: Any) -> None:
    if span is not None and span.is_recording():
        _scrub_span_urls(span)


async def _async_client_response_hook(span: Span, request: Any, response: Any) -> None:
    _client_response_hook(span, request, response)


def instrument_app(app: FastAPI, cfg: TelemetryConfig) -> None:
    """FastAPIInstrumentor.instrument_app (server span, excluded urls /healthz,/readyz) and
    HTTPXClientInstrumentor().instrument() (CLIENT spans + traceparent injection upstream).

    Uses the providers of the most recent OtelTelemetry (falls back to OTel globals when
    telemetry is disabled, i.e. effectively no-op). URL/query attributes are scrubbed by hooks so
    query-string credentials never reach a span. httpx instrumentation patches clients created
    AFTER this call; call it before building the proxy's AsyncClient, or use
    `instrument_httpx_client` for an existing one."""
    if not cfg.enabled:
        return
    active = _ACTIVE
    tp = active.tracer_provider if active is not None else None
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tp,
        excluded_urls=_EXCLUDED_URLS,
        server_request_hook=_server_request_hook,
    )
    httpx_inst = HTTPXClientInstrumentor()
    if httpx_inst.is_instrumented_by_opentelemetry:
        httpx_inst.uninstrument()
    httpx_inst.instrument(
        tracer_provider=tp,
        request_hook=_client_request_hook,
        response_hook=_client_response_hook,
        async_request_hook=_async_client_request_hook,
        async_response_hook=_async_client_response_hook,
    )


def instrument_httpx_client(client: httpx.AsyncClient | httpx.Client) -> None:
    """Instrument one already-created httpx client (CLIENT spans + traceparent injection)."""
    active = _ACTIVE
    is_async = isinstance(client, httpx.AsyncClient)
    HTTPXClientInstrumentor.instrument_client(
        client,
        tracer_provider=active.tracer_provider if active is not None else None,
        request_hook=_async_client_request_hook if is_async else _client_request_hook,
        response_hook=_async_client_response_hook if is_async else _client_response_hook,
    )


def uninstrument(app: FastAPI | None = None) -> None:
    """Undo instrument_app (tests / shutdown)."""
    if app is not None:
        FastAPIInstrumentor.uninstrument_app(app)
    inst = HTTPXClientInstrumentor()
    if inst.is_instrumented_by_opentelemetry:
        inst.uninstrument()


__all__ = [
    "OtelTelemetry",
    "instrument_app",
    "instrument_httpx_client",
    "setup_telemetry",
    "uninstrument",
]
