from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_gateway.config import TelemetryConfig
from ai_gateway.observability.logging import shutdown_logging
from ai_gateway.observability.otel import OtelTelemetry, uninstrument

BEARER = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJlLXZhbHVl"
API_KEY = "sk-ant-api03-AbCdEf0123456789XyZ"
SECRETS = (BEARER, API_KEY, "hunter2-pass")


@dataclass
class Rig:
    tel: OtelTelemetry
    spans: InMemorySpanExporter
    metrics: InMemoryMetricReader
    logs: InMemoryLogRecordExporter
    stdout: io.StringIO

    def metric(self, name: str) -> list[tuple[dict[str, Any], Any]]:
        """(attributes, point) for every data point of metric `name`."""
        data = self.metrics.get_metrics_data()
        out: list[tuple[dict[str, Any], Any]] = []
        for rm in data.resource_metrics if data else []:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    if m.name == name:
                        out.extend((dict(p.attributes or {}), p) for p in m.data.data_points)
        return out

    def metric_names(self) -> set[str]:
        data = self.metrics.get_metrics_data()
        return {
            m.name
            for rm in (data.resource_metrics if data else [])
            for sm in rm.scope_metrics
            for m in sm.metrics
        }

    def log_records(self, scope: str | None = None) -> list[tuple[str, dict[str, Any], LogRecord]]:
        out: list[tuple[str, dict[str, Any], LogRecord]] = []
        for ld in self.logs.get_finished_logs():
            name = ld.instrumentation_scope.name if ld.instrumentation_scope else ""
            if scope is None or name == scope:
                rec = ld.log_record
                out.append((name, dict(rec.attributes or {}), rec))
        return out

    def everything(self) -> str:
        """All captured telemetry as one string, for 'secret never appears' assertions."""
        parts = [self.stdout.getvalue()]
        for s in self.spans.get_finished_spans():
            parts.append(s.to_json())
        for _, attrs, rec in self.log_records():
            parts.append(repr(attrs) + repr(rec.body))
        return "\n".join(parts)


@pytest.fixture
def rig() -> Iterator[Rig]:
    cfg = TelemetryConfig()
    spans = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    logs = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
    out = io.StringIO()
    tel = OtelTelemetry(cfg, span_exporter=spans, metric_reader=reader, log_exporter=logs, stdout=out)
    try:
        yield Rig(tel, spans, reader, logs, out)
    finally:
        tel._flush_and_stop()
        shutdown_logging()
        uninstrument()


def attrs_of(span: Any) -> dict[str, Any]:
    return dict(span.attributes or {})
