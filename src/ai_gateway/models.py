"""Shared domain types. Imported by every module; imports nothing from the package.

OWNER: core. Changing a field here is a contract change - announce it to all owners.
These are plain dataclasses/enums (no I/O, no pydantic) so every layer can use them cheaply.
"""

from __future__ import annotations

import enum
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class UpstreamKind(enum.StrEnum):
    """What protocol an upstream speaks. Selects the proxy adapter and the policy grant type."""

    HTTP = "http"
    MCP = "mcp"
    LLM = "llm"


class Phase(enum.StrEnum):
    """Rate-limit phase reached by a request (also the metric attribute value)."""

    ALLOW = "allow"  # below delay threshold
    DELAY = "delay"  # phase 1: admitted after a bounded async wait
    REJECT = "reject"  # phase 2: 429 + Retry-After + RateLimit-* headers
    BLOCK = "block"  # phase 3: client in timed cooldown, rejected before any work


class LimitScope(enum.StrEnum):
    REQUESTS = "requests"
    TOKENS = "tokens"
    IP = "ip"  # pre-auth per-IP limiter


class DecisionReason(enum.StrEnum):
    """Authorization outcome codes (audit + metric attribute; never shown verbatim to clients)."""

    ALLOWED = "allowed"
    NO_ROUTE = "no_route"  # path matches no upstream prefix
    NO_MATCHING_RULE = "no_matching_rule"  # default deny
    INSUFFICIENT_SCOPE = "insufficient_scope"  # a rule would match, principal lacks scopes
    EXPLICIT_DENY = "explicit_deny"  # a deny rule matched (always wins)
    MALFORMED_REQUEST = "malformed_request"  # body unparsable, cannot classify -> deny


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated caller. Built ONLY by auth.TokenValidator. Never holds the raw token."""

    client_id: str  # stable id used for rate limits, metrics, audit
    subject: str  # JWT `sub`
    issuer: str
    scopes: frozenset[str]
    audience: tuple[str, ...] = ()
    token_id: str | None = None  # jti, for audit correlation
    expires_at: float = 0.0  # epoch seconds
    idp: str = ""  # IdPConfig.name that validated the token
    is_client_credentials: bool = False  # machine client (no end-user) heuristic: sub == client_id
    claims: Mapping[str, Any] = field(default_factory=dict)  # allow-listed, non-sensitive claims


@dataclass(frozen=True, slots=True)
class RouteMatch:
    """Result of mapping a public request path to a configured upstream."""

    upstream: str  # key in PolicyConfig.upstreams
    kind: UpstreamKind
    prefix: str  # public prefix that matched, e.g. "/billing"
    relative_path: str  # path after prefix, always starts with "/", e.g. "/invoices/42"


@dataclass(frozen=True, slots=True)
class Resource:
    """One authorizable thing a request touches. A request may carry several (MCP batches)."""

    kind: UpstreamKind
    upstream: str
    method: str  # HTTP method, upper-case
    path: str  # relative_path
    mcp_method: str | None = None  # JSON-RPC method, e.g. "tools/call"
    tool: str | None = None  # MCP tool name (tools/call)
    model: str | None = None  # LLM model id from request body


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: DecisionReason
    resource: Resource | None = None
    rule_id: str | None = None  # rule that produced the allow / explicit deny
    matched_scopes: frozenset[str] = frozenset()
    missing_scopes_hint: tuple[str, ...] = ()  # audit only; NEVER returned to the client
    max_tokens_per_request: int | None = None  # LLM grant cap, if any


@dataclass(frozen=True, slots=True)
class LimitDecision:
    """Outcome of a rate-limit check. `allowed` is True for ALLOW and DELAY."""

    phase: Phase
    scope: LimitScope
    client_id: str
    limit: int  # quota per window
    remaining: int  # never negative
    reset_s: float  # seconds until the window frees capacity
    usage_ratio: float = 0.0  # attempts / quota after counting this request
    delay_s: float = 0.0  # DELAY: how long apply_delay() will wait
    retry_after_s: float = 0.0  # REJECT/BLOCK: Retry-After value
    window_s: float = 60.0
    reason: str = ""
    slot: str | None = None  # opaque handle for a held delay-concurrency slot (limiter-internal)

    @property
    def allowed(self) -> bool:
        return self.phase in (Phase.ALLOW, Phase.DELAY)

    def headers(self) -> dict[str, str]:
        """RateLimit-Limit/Remaining/Reset (+ Policy) always; Retry-After when rejected."""
        h = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(max(self.remaining, 0)),
            "RateLimit-Reset": str(max(math.ceil(self.reset_s), 0)),
            "RateLimit-Policy": f"{self.limit};w={int(self.window_s)}",
        }
        if not self.allowed:
            h["Retry-After"] = str(max(math.ceil(self.retry_after_s), 1))
        return h


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """LLM token usage. `estimated=True` means pre-flight estimate (upstream gave no usage)."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    provider: str | None = None
    estimated: bool = False

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class RequestContext:
    """Mutable per-request state, created by the request-id stage, filled stage by stage.

    Stored in a ContextVar (see context.py) so loggers/telemetry can read ids without plumbing.
    """

    request_id: str
    method: str
    path: str
    client_ip: str = ""
    trace_id: str = ""
    started_at: float = field(default_factory=time.monotonic)
    principal: Principal | None = None
    limit_key: str = ""  # rate-limit identity: "{idp}:{client_id}" (+ "|{sub}" for end-user tokens)
    client_label: str = "anonymous"  # metric-safe: registered client_id | "unregistered" | "anonymous"
    route: RouteMatch | None = None
    resources: list[Resource] = field(default_factory=list)
    policy: PolicyDecision | None = None
    limit: LimitDecision | None = None
    token_limit: LimitDecision | None = None
    estimated_tokens: int = 0
    usage: UsageRecord | None = None
    status_code: int | None = None
    error_code: str | None = None

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at
