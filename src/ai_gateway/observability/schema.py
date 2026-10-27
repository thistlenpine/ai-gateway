"""Canonical telemetry names. OWNER: observability. Everyone else imports these constants
instead of typing strings, so dashboards/alerts never drift. Do not rename without a DESIGN.md
update."""

from __future__ import annotations

from typing import Final

# ---- span names (one per pipeline stage; all children of the FastAPI server span)
SPAN_REQUEST: Final = "gateway.request"  # created by instrumentation-fastapi (server span)
SPAN_AUTHN: Final = "gateway.authn"
SPAN_JWKS_REFRESH: Final = "gateway.authn.jwks_refresh"
SPAN_AUTHZ: Final = "gateway.authz"
SPAN_RATELIMIT: Final = "gateway.ratelimit"
SPAN_RATELIMIT_DELAY: Final = "gateway.ratelimit.delay"
SPAN_PROXY: Final = "gateway.proxy"  # parent of the httpx CLIENT span (traceparent injected)
SPAN_ACCOUNTING: Final = "gateway.accounting"

# ---- metric names (OTel instruments; units in comments)
M_REQUESTS: Final = "gateway.requests"  # Counter {request}
M_REQUEST_DURATION: Final = "gateway.request.duration"  # Histogram s
M_ACTIVE: Final = "gateway.requests.active"  # UpDownCounter {request}
M_RL_DECISIONS: Final = "gateway.ratelimit.decisions"  # Counter {decision}
M_RL_DELAY: Final = "gateway.ratelimit.delay.duration"  # Histogram s
M_AUTH_FAILURES: Final = "gateway.auth.failures"  # Counter {failure}
M_AUTHZ_DECISIONS: Final = "gateway.authz.decisions"  # Counter {decision}
M_UPSTREAM_DURATION: Final = "gateway.upstream.duration"  # Histogram s
M_UPSTREAM_ERRORS: Final = "gateway.upstream.errors"  # Counter {error}
M_LLM_TOKENS: Final = "gateway.llm.tokens"  # Counter {token}
M_JWKS_REFRESH: Final = "gateway.jwks.refreshes"  # Counter {refresh}

# ---- attribute keys (spans, metrics, logs, audit share these)
A_REQUEST_ID: Final = "gateway.request_id"
A_CLIENT_ID: Final = "gateway.client_id"  # metrics: registered id | "unregistered" | "anonymous"
A_SUBJECT: Final = "enduser.id"
A_ISSUER: Final = "gateway.idp.issuer"
A_UPSTREAM: Final = "gateway.upstream"
A_UPSTREAM_KIND: Final = "gateway.upstream.kind"
A_HTTP_METHOD: Final = "http.request.method"
A_HTTP_STATUS: Final = "http.response.status_code"
A_AUTH_REASON: Final = "gateway.auth.failure_reason"
A_AUTHZ_DECISION: Final = "gateway.authz.decision"  # allow | deny
A_AUTHZ_REASON: Final = "gateway.authz.reason"
A_RULE_ID: Final = "gateway.policy.rule_id"
A_RL_PHASE: Final = "gateway.ratelimit.phase"  # allow | delay | reject | block
A_RL_SCOPE: Final = "gateway.ratelimit.scope"  # requests | tokens | ip
A_RL_RATIO: Final = "gateway.ratelimit.usage_ratio"
A_ERROR_TYPE: Final = "error.type"
A_LLM_SYSTEM: Final = "gen_ai.system"
A_LLM_MODEL: Final = "gen_ai.request.model"
A_LLM_TOKEN_TYPE: Final = "gen_ai.token.type"  # input | output
A_MCP_TOOL: Final = "gateway.mcp.tool"
A_MCP_METHOD: Final = "mcp.method.name"
A_JWKS_OUTCOME: Final = "gateway.jwks.outcome"  # hit | refreshed | stale | failed

# ---- log record fields (JSON; see observability/logging.py)
L_TIMESTAMP: Final = "timestamp"
L_LEVEL: Final = "level"
L_MESSAGE: Final = "message"
L_LOGGER: Final = "logger"
L_TRACE_ID: Final = "trace_id"
L_SPAN_ID: Final = "span_id"
L_STREAM: Final = "stream"  # "app" | "audit"
L_EVENT: Final = "event"  # stable machine name, e.g. "authn.failed"
L_REQUEST_ID: Final = "request_id"
L_CLIENT_ID: Final = "client_id"

# Attribute keys that must never appear in any log/span/metric/audit payload.
REDACT_KEYS: Final = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        "api_key",
        "password",
    }
)
