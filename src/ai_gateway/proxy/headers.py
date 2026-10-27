"""Header hygiene for proxying. OWNER: proxy."""

from __future__ import annotations

import re

from opentelemetry import propagate

from ai_gateway.config import UpstreamConfig
from ai_gateway.models import RequestContext

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

# Never forwarded upstream: the caller's credentials, and anything that claims to be identity or
# client-address information set by a proxy (the client must not be able to forge those).
_DROP_REQUEST = frozenset(
    {
        "authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "x-auth-token",
        "forwarded",
        "via",
        "x-real-ip",
        "x-client-ip",
        "true-client-ip",
        "cf-connecting-ip",
        "x-original-url",
        "x-original-uri",
        "x-rewrite-url",
        "x-http-method-override",
        "x-http-method",
        "x-method-override",
        "expect",
        "accept-encoding",  # re-set by the client per upstream kind
        "traceparent",
        "tracestate",
        "baggage",
        "x-request-id",
    }
)
_DROP_PREFIXES = ("x-gateway-", "x-forwarded-", "x-envoy-", "x-amzn-trace", "proxy-")

_BAD_CHARS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")  # control chars incl. CR/LF (HTAB allowed)
_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_NON_PRINTABLE = re.compile(r"[^\x20-\x7e]")


def _connection_tokens(headers: list[tuple[str, str]]) -> set[str]:
    out: set[str] = set()
    for name, value in headers:
        if name.lower() == "connection":
            out.update(t.strip().lower() for t in value.split(",") if t.strip())
    return out


def _ascii(value: str) -> str:
    return _NON_PRINTABLE.sub("", value)


def build_upstream_headers(
    inbound: list[tuple[str, str]],
    upstream: UpstreamConfig,
    ctx: RequestContext,
    *,
    accept_encoding: str | None = None,
    content_length: str | None = None,
) -> list[tuple[str, str]]:
    """Drop hop-by-hop + inbound Authorization/Cookie + anything x-gateway-*/x-forwarded-*; add
    upstream credential (UpstreamCredential.resolve(); raises ConfigError if the env var is
    unset -> fail closed), X-Request-ID, and (if forward_identity_headers) X-Gateway-Client-Id /
    X-Gateway-Subject. Keeps Mcp-Session-Id, Accept, Content-Type, Last-Event-ID,
    anthropic-version etc. traceparent/tracestate are injected from the active span."""
    conn = _connection_tokens(inbound)
    cred_header = upstream.credential.header.lower() if upstream.credential else None
    out: list[tuple[str, str]] = []
    for name, value in inbound:
        low = name.lower()
        if (
            low in HOP_BY_HOP
            or low in conn
            or low in _DROP_REQUEST
            or low.startswith(_DROP_PREFIXES)
            or low == cred_header
            or not _TOKEN.match(name)
            or _BAD_CHARS.search(value)
        ):
            continue
        out.append((name, value))

    if upstream.credential is not None:
        # resolve() raises ConfigError when the env var is missing: never send without it.
        out.append((upstream.credential.header, upstream.credential.resolve()))
    out.append(("X-Request-ID", _ascii(ctx.request_id)))
    if upstream.forward_identity_headers and ctx.principal is not None:
        out.append(("X-Gateway-Client-Id", _ascii(ctx.principal.client_id)))
        out.append(("X-Gateway-Subject", _ascii(ctx.principal.subject)))
    if accept_encoding is not None:
        out.append(("Accept-Encoding", accept_encoding))
    if content_length is not None:
        out.append(("Content-Length", content_length))

    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    for key in ("traceparent",):  # tracestate is vendor data from the caller: never relayed
        if key in carrier:
            out.append((key, carrier[key]))
    return out


# Ambient-authority / fingerprinting headers one upstream must not be able to set on the gateway's
# origin (cookies are scoped to the gateway host and would be honoured for every other upstream).
_DROP_RESPONSE = frozenset({"set-cookie", "set-cookie2", "server", "x-powered-by", "via"})


def _latin1(text: str) -> bool:
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def filter_response_headers(
    headers: list[tuple[str, str]], *, decoded: bool = False
) -> list[tuple[str, str]]:
    """Drop hop-by-hop and content-length (body may be re-framed/rewritten); also
    content-encoding when the body was decoded by httpx (`decoded=True`)."""
    conn = _connection_tokens(headers)
    out: list[tuple[str, str]] = []
    for name, value in headers:
        low = name.lower()
        if low in HOP_BY_HOP or low in conn or (decoded and low == "content-encoding"):
            continue
        if low in _DROP_RESPONSE or _BAD_CHARS.search(value) or not _latin1(name + value):
            continue
        out.append((name, value))
    return out


def get_header(headers: list[tuple[str, str]], name: str) -> str | None:
    low = name.lower()
    for k, v in headers:
        if k.lower() == low:
            return v
    return None
