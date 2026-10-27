"""Observability package (OTel only). Public surface: Telemetry, NullTelemetry, setup_telemetry."""

from ai_gateway.observability.audit import AuditEvent, AuditType
from ai_gateway.observability.otel import instrument_app, setup_telemetry
from ai_gateway.observability.telemetry import NullTelemetry, Telemetry

__all__ = [
    "AuditEvent",
    "AuditType",
    "NullTelemetry",
    "Telemetry",
    "instrument_app",
    "setup_telemetry",
]
