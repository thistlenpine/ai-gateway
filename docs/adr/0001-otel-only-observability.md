# ADR 0001: OpenTelemetry only for observability

Status: accepted

## Context
The gateway needs traces, metrics, logs and an audit stream. Supporting Prometheus scrape endpoints plus OTLP doubles code paths, adds an unauthenticated listener, and invites label-cardinality drift.

## Decision
Use the OpenTelemetry API/SDK for all three signals, exported via OTLP (grpc or http/protobuf) to a collector. No Prometheus client, no `/metrics`. Modules depend on a `Telemetry` facade (`observability/telemetry.py`) with a fully working `NullTelemetry`; only `observability/otel.py` touches the SDK. Names live in `observability/schema.py`. Audit events go through the OTel Logs API under a distinct scope (`ai_gateway.audit`).

## Consequences
- Operators needing Prometheus use the collector's prometheus exporter.
- One propagation story (W3C traceparent) and one config surface.
- OTel Logs/metrics SDK APIs are younger; pin versions and isolate in one module.
- Telemetry failures must never block requests (batch processors, drop on overflow).
