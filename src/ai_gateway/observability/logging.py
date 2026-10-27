"""Structured JSON logging with trace/correlation ids. OWNER: observability."""

from __future__ import annotations

import copy
import json
import logging
import sys
import warnings
from datetime import UTC, datetime
from typing import IO, Any, Final

from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

from ai_gateway.config import TelemetryConfig
from ai_gateway.context import get_ctx
from ai_gateway.observability import schema
from ai_gateway.observability.redact import redact_value, scrub_text

AUDIT_LOGGER_NAME: Final = "ai_gateway.audit"
_HANDLER_MARK: Final = "_ai_gateway_handler"
_MAX_EXC_CHARS: Final = 8192

# Attributes every LogRecord carries; everything else came from `extra=`.
_STD_ATTRS: Final = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"message", "asctime", "taskName", "audit"}
)
_RESERVED_OUT: Final = frozenset(
    {schema.L_TIMESTAMP, schema.L_LEVEL, schema.L_MESSAGE, schema.L_LOGGER, schema.L_STREAM}
)


def current_trace_ids() -> tuple[str, str]:
    """(trace_id, span_id) of the active valid span as lowercase hex, else ("", "")."""
    sc = trace.get_current_span().get_span_context()
    if not sc.is_valid:
        return "", ""
    return f"{sc.trace_id:032x}", f"{sc.span_id:016x}"


def correlation_fields() -> dict[str, str]:
    """request_id / client_id (from RequestContext) and trace_id / span_id (active span)."""
    out: dict[str, str] = {}
    ctx = get_ctx()
    trace_id, span_id = current_trace_ids()
    if ctx is not None:
        out[schema.L_REQUEST_ID] = ctx.request_id
        out[schema.L_CLIENT_ID] = ctx.client_label
        trace_id = trace_id or ctx.trace_id
    if trace_id:
        out[schema.L_TRACE_ID] = trace_id
    if span_id:
        out[schema.L_SPAN_ID] = span_id
    return out


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """User supplied fields (redacted); `record.audit` payload is flattened in."""
    out: dict[str, Any] = {}
    audit = record.__dict__.get("audit")
    if isinstance(audit, dict):
        out.update({str(k): redact_value(k, v) for k, v in audit.items()})
    for k, v in record.__dict__.items():
        if k not in _STD_ATTRS and not k.startswith("_"):
            out[k] = redact_value(k, v)
    return out


def _exception_text(record: logging.LogRecord, fmt: logging.Formatter) -> str | None:
    if record.exc_info and record.exc_info[0] is not None:
        text = fmt.formatException(record.exc_info)
    elif record.exc_text:
        text = record.exc_text
    else:
        return None
    return scrub_text(text)[:_MAX_EXC_CHARS]


def _message(record: logging.LogRecord) -> str:
    try:
        return scrub_text(record.getMessage())
    except Exception:
        return scrub_text(str(record.msg))


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Fields per schema.L_*; merges `extra` keys; adds request_id /
    client_id from context.get_ctx() and trace_id/span_id from the active OTel span.
    Redacts schema.REDACT_KEYS (case-insensitive) anywhere in extra and scrubs credential-looking
    text from the message and exception text."""

    def __init__(self, *, stream: str = "app") -> None:
        super().__init__()
        self._stream = stream

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            schema.L_TIMESTAMP: datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            schema.L_LEVEL: record.levelname,
            schema.L_MESSAGE: _message(record),
            schema.L_LOGGER: record.name,
            schema.L_STREAM: self._stream,
            **correlation_fields(),
        }
        for k, v in _extras(record).items():
            if k in _RESERVED_OUT or v in ("", None):
                continue
            out[k] = v
        exc = _exception_text(record, self)
        if exc:
            out["exception"] = exc
        return json.dumps(out, default=str, ensure_ascii=False)


def _otel_value(value: Any) -> Any:
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple) and all(isinstance(v, str | bool | int | float) for v in value):
        kinds = {type(v) for v in value}
        if len(kinds) <= 1:
            return list(value)
    return json.dumps(value, default=str, ensure_ascii=False)


class RedactingOtelHandler(LoggingHandler):
    """OTel LoggingHandler that exports a sanitized copy of each record: scrubbed message and
    exception text, redacted extras (flattened to OTel attribute types), correlation fields."""

    def __init__(self, level: int, logger_provider: LoggerProvider, *, stream: str) -> None:
        with warnings.catch_warnings():  # SDK deprecates its handler; the replacement is not a dependency
            warnings.simplefilter("ignore", DeprecationWarning)
            super().__init__(level=level, logger_provider=logger_provider)
        self._stream = stream
        self._fmt = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("opentelemetry"):
            return  # never export the exporter's own diagnostics (feedback loop)
        safe = copy.copy(record)
        safe.msg = _message(record)
        safe.args = None
        exc = _exception_text(record, self._fmt)
        safe.exc_info = None
        safe.exc_text = None
        for k in [k for k in safe.__dict__ if k not in _STD_ATTRS or k == "audit"]:
            del safe.__dict__[k]
        attrs: dict[str, Any] = {
            **{k: v for k, v in _extras(record).items() if v not in ("", None)},
            **{k: v for k, v in correlation_fields().items() if k not in ("trace_id", "span_id")},
            schema.L_STREAM: self._stream,
        }
        if exc:
            attrs["exception.stacktrace"] = exc
        for k, v in attrs.items():
            safe.__dict__[k] = _otel_value(v)
        super().emit(safe)


def remove_marked_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        if getattr(h, _HANDLER_MARK, False):
            logger.removeHandler(h)
            h.close()


def mark_handler(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _HANDLER_MARK, True)
    return handler


def configure_logging(
    cfg: TelemetryConfig,
    *,
    logger_provider: LoggerProvider | None = None,
    stream: IO[str] | None = None,
) -> None:
    """Root logger: JSON to stdout (if cfg.log_stdout_json) + OTel LoggingHandler (OTLP logs).
    The `ai_gateway.audit` logger is configured separately by the audit sink and does not
    propagate to root (prevents duplicate audit lines). Idempotent: previously installed
    gateway handlers are replaced."""
    root = logging.getLogger()
    remove_marked_handlers(root)
    level = logging.getLevelNamesMapping().get(cfg.log_level.upper(), logging.INFO)
    root.setLevel(level)
    for noisy in ("httpx", "httpcore"):  # httpx logs full request URLs at INFO (query-string credentials)
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
    if cfg.log_stdout_json:
        sh = logging.StreamHandler(stream if stream is not None else sys.stdout)
        sh.setFormatter(JsonFormatter())
        root.addHandler(mark_handler(sh))
    if logger_provider is not None:
        root.addHandler(mark_handler(RedactingOtelHandler(level, logger_provider, stream="app")))


def shutdown_logging() -> None:
    """Remove every handler installed by this module (root and audit loggers)."""
    remove_marked_handlers(logging.getLogger())
    remove_marked_handlers(logging.getLogger(AUDIT_LOGGER_NAME))
