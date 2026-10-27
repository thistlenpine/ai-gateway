"""httpx upstream client + Proxy entry point. OWNER: proxy."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from urllib.parse import quote, urlsplit

import httpx

from ai_gateway.config import ConfigError, DefaultsConfig, PolicyConfig, UpstreamConfig
from ai_gateway.errors import (
    BadRequestError,
    GatewayError,
    PayloadTooLargeError,
    UpstreamError,
    UpstreamTimeoutError,
)
from ai_gateway.models import RouteMatch, UpstreamKind, UsageRecord
from ai_gateway.observability import schema
from ai_gateway.observability.telemetry import Telemetry
from ai_gateway.proxy.base import Adapter, ProxyRequest, ProxyResponse, UsageCallback
from ai_gateway.proxy.headers import build_upstream_headers, filter_response_headers, get_header
from ai_gateway.proxy.llm import LlmAdapter
from ai_gateway.proxy.mcp import McpAdapter
from ai_gateway.proxy.passthrough import HttpAdapter

log = logging.getLogger(__name__)

DEFAULT_MAX_REQUEST_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024 * 1024  # non-SSE bodies; event streams are unbounded

_METHOD = re.compile(r"^[A-Z]{3,10}$")
_FORBIDDEN_METHODS = frozenset({"CONNECT", "TRACE"})
_QUERY_BAD = re.compile(r"[\x00-\x20\x7f#\\]")
_PATH_SAFE = "/:@!$&'()*+,=-._~"


def build_http_client(defaults: DefaultsConfig) -> httpx.AsyncClient:
    """Shared AsyncClient: limits (max_connections=200, max_keepalive=50),
    timeout=httpx.Timeout(defaults.upstream_timeout_s, connect=defaults.upstream_connect_timeout_s),
    follow_redirects=False, trust_env=False. HTTP/2 is negotiated when the `h2` package is
    installed. Same factory (separate instance) is used for JWKS."""
    try:
        import h2  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

        http2 = True
    except ImportError:
        http2 = False
    return httpx.AsyncClient(
        timeout=httpx.Timeout(defaults.upstream_timeout_s, connect=defaults.upstream_connect_timeout_s),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        follow_redirects=False,
        trust_env=False,
        http2=http2,
    )


def build_upstream_url(upstream: UpstreamConfig, route: RouteMatch, query: str) -> httpx.URL:
    """base_url + path + query. The destination is fixed by config; the client only influences
    the (re-encoded) path and query, never scheme/host/port."""
    rel = route.relative_path
    if not rel.startswith("/") or rel.startswith("//") or "\\" in rel or "\x00" in rel:
        raise BadRequestError("Invalid path")
    if any(seg in ("..", ".") for seg in rel.split("/")):
        raise BadRequestError("Invalid path")
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in rel):
        raise BadRequestError("Invalid path")
    if rel == "/":
        # Prefix root: forward exactly base_url (+ prefix if kept); never invent a trailing slash,
        # `/mcp` and `/mcp/` are different resources on most servers (SEC-07b).
        path = "" if upstream.strip_prefix else upstream.prefix.rstrip("/")
    else:
        path = rel if upstream.strip_prefix else upstream.prefix.rstrip("/") + rel
    if query and _QUERY_BAD.search(query):
        raise BadRequestError("Invalid query string")

    base = httpx.URL(upstream.base_url)
    if base.scheme not in ("http", "https") or not base.host:
        raise UpstreamError("Upstream misconfigured")
    base_path = base.raw_path.decode("ascii").split("?", 1)[0].rstrip("/")
    full = (base_path + quote(path, safe=_PATH_SAFE)) or "/"
    url = base.copy_with(raw_path=(full + (f"?{query}" if query else "")).encode("ascii"))
    # Defence in depth: nothing client-supplied may have changed the destination.
    if (url.scheme, url.host, url.port) != (base.scheme, base.host, base.port) or url.userinfo:
        raise BadRequestError("Invalid path")
    return url


def _rewrite_location(
    headers: list[tuple[str, str]], upstream: UpstreamConfig, route: RouteMatch
) -> list[tuple[str, str]]:
    """Redirects are never followed, but the caller must not learn internal origins (SEC-07):
    a Location on the upstream's own origin becomes a gateway-relative path under the public
    prefix; any other absolute Location is dropped."""
    if not any(k.lower() == "location" for k, _ in headers):
        return headers
    base = httpx.URL(upstream.base_url)
    base_path = base.raw_path.decode("ascii").split("?", 1)[0].rstrip("/")
    out: list[tuple[str, str]] = []
    for name, value in headers:
        if name.lower() != "location":
            out.append((name, value))
            continue
        parts = urlsplit(value)
        if parts.scheme or parts.netloc:
            try:
                loc = httpx.URL(value)
            except httpx.InvalidURL:
                continue
            same = (loc.scheme, loc.host, loc.port) == (base.scheme, base.host, base.port)
            if not same:
                continue  # foreign / internal address: not ours to relay
        elif not parts.path.startswith("/") or parts.path.startswith("//"):
            out.append((name, value))  # relative reference: resolves against the public URL
            continue
        path = parts.path
        if base_path and (path == base_path or path.startswith(base_path + "/")):
            path = path[len(base_path) :]
        prefix = route.prefix.rstrip("/") if upstream.strip_prefix else ""
        new = (prefix + path) or "/"
        if parts.query:
            new += "?" + parts.query
        out.append((name, new))
    return out


def _bounded_request_stream(source: AsyncIterator[bytes], limit: int) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        total = 0
        async for chunk in source:
            total += len(chunk)
            if total > limit:
                raise PayloadTooLargeError()
            yield chunk

    return gen()


async def _single(data: bytes) -> AsyncIterator[bytes]:
    if data:
        yield data


class Proxy:
    """Forwards one request to the upstream chosen by `route` and returns a streaming response.

    forward():
      - URL = upstream.base_url + (relative_path if strip_prefix else prefix+relative_path) + query
      - headers via headers.build_upstream_headers; body streamed (http) or bytes (mcp/llm)
      - client.send(request, stream=True) inside SPAN_PROXY; traceparent injected
      - no automatic retries (non-idempotent); httpx.TimeoutException -> UpstreamTimeoutError,
        other httpx.HTTPError -> UpstreamError; both call telemetry.record_upstream(...)
      - never follows redirects; 3xx passed to the client as-is
      - returns adapter.process_response() output; caller wires tool_allowed/on_usage via
        forward(..., tool_allowed=..., on_usage=...)
    """

    def __init__(
        self,
        policy: PolicyConfig,
        http: httpx.AsyncClient,
        telemetry: Telemetry,
        defaults: DefaultsConfig,
        *,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self._policy = policy
        self._http = http
        self._telemetry = telemetry
        self._defaults = defaults
        self._max_request = max_request_bytes
        self._max_response = max_response_bytes
        self._adapters: dict[UpstreamKind, Adapter] = {
            UpstreamKind.HTTP: HttpAdapter(),
            UpstreamKind.MCP: McpAdapter(),
            UpstreamKind.LLM: LlmAdapter(),
        }

    def adapter(self, kind: UpstreamKind) -> Adapter:
        """HttpAdapter / McpAdapter / LlmAdapter singleton for `kind`."""
        return self._adapters[kind]

    async def forward(
        self,
        route: RouteMatch,
        req: ProxyRequest,
        *,
        tool_allowed: Callable[[str], bool] | None = None,
        on_usage: UsageCallback | None = None,
    ) -> ProxyResponse:
        upstream = self._policy.upstreams.get(route.upstream)
        if upstream is None or upstream.kind is not route.kind:
            raise UpstreamError("Upstream misconfigured")
        adapter = self.adapter(upstream.kind)
        method = req.method.upper()
        if not _METHOD.match(method) or method in _FORBIDDEN_METHODS:
            raise BadRequestError("Method not allowed")

        req = adapter.prepare(upstream, req)
        url = build_upstream_url(upstream, route, req.query)

        body = req.body
        content_length: str | None = None
        if isinstance(body, bytes | bytearray):
            if len(body) > self._max_request:
                raise PayloadTooLargeError()
        elif body is not None:
            declared = get_header(req.headers, "content-length")
            if declared is not None:
                if not declared.isascii() or not declared.isdigit():
                    raise BadRequestError("Invalid Content-Length")
                if int(declared) > self._max_request:
                    raise PayloadTooLargeError()
                content_length = str(int(declared))
            body = _bounded_request_stream(body, self._max_request)

        decode = upstream.kind is not UpstreamKind.HTTP  # mcp/llm bodies are inspected
        try:
            headers = build_upstream_headers(
                req.headers,
                upstream,
                req.ctx,
                accept_encoding="identity" if decode else get_header(req.headers, "accept-encoding"),
                content_length=content_length,
            )
        except ConfigError:
            log.error("upstream credential unavailable", extra={"event": "proxy.credential_missing"})
            self._telemetry.record_upstream(
                upstream=route.upstream,
                kind=str(upstream.kind),
                duration_s=0.0,
                status=None,
                error_type="credential_missing",
            )
            raise UpstreamError(
                "Upstream credential unavailable", code="upstream_credential_missing"
            ) from None

        timeout = httpx.Timeout(
            upstream.timeout_s or self._defaults.upstream_timeout_s,
            connect=self._defaults.upstream_connect_timeout_s,
        )
        attrs = {
            schema.A_UPSTREAM: route.upstream,
            schema.A_UPSTREAM_KIND: str(upstream.kind),
        }
        started = time.monotonic()
        try:
            with self._telemetry.span(schema.SPAN_PROXY, attrs):
                request = self._http.build_request(
                    method, url, headers=headers, content=body, timeout=timeout
                )
                resp = await self._http.send(request, stream=True)
        except GatewayError:
            raise
        except httpx.TimeoutException:
            self._record_error(route, upstream, started, "timeout")
            raise UpstreamTimeoutError() from None
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            self._record_error(route, upstream, started, type(exc).__name__)
            raise UpstreamError() from None

        self._telemetry.record_upstream(
            upstream=route.upstream,
            kind=str(upstream.kind),
            duration_s=time.monotonic() - started,
            status=resp.status_code,
        )

        declared_len = resp.headers.get("content-length", "")
        is_stream = resp.headers.get("content-type", "").lower().startswith("text/event-stream")
        if not is_stream and declared_len.isdigit() and int(declared_len) > self._max_response:
            await resp.aclose()
            raise UpstreamError("Upstream response too large")

        out_headers = _rewrite_location(
            filter_response_headers(list(resp.headers.multi_items()), decoded=decode), upstream, route
        )
        proxy_resp = ProxyResponse(
            status=resp.status_code,
            headers=out_headers,
            body=self._body(resp, route, upstream, decode=decode, bounded=not is_stream),
            is_stream=is_stream,
            _close=resp.aclose,
        )
        cb = self._wrap_usage(on_usage, upstream, req) if upstream.kind is UpstreamKind.LLM else on_usage
        return adapter.process_response(proxy_resp, tool_allowed=tool_allowed, on_usage=cb)

    def _record_error(self, route: RouteMatch, upstream: UpstreamConfig, started: float, error: str) -> None:
        self._telemetry.record_upstream(
            upstream=route.upstream,
            kind=str(upstream.kind),
            duration_s=time.monotonic() - started,
            status=None,
            error_type=error,
        )

    async def _body(
        self,
        resp: httpx.Response,
        route: RouteMatch,
        upstream: UpstreamConfig,
        *,
        decode: bool,
        bounded: bool,
    ) -> AsyncIterator[bytes]:
        total = 0
        try:
            if resp.is_stream_consumed:  # in-memory response (test transports); nothing to stream
                source: AsyncIterator[bytes] = _single(resp.content)
            else:
                source = resp.aiter_bytes() if decode else resp.aiter_raw()
            async for chunk in source:
                total += len(chunk)
                if bounded and total > self._max_response:
                    raise UpstreamError("Upstream response too large")
                yield chunk
        except httpx.TimeoutException:
            self._record_error(route, upstream, time.monotonic(), "timeout")
            raise UpstreamTimeoutError() from None
        except httpx.HTTPError as exc:
            self._record_error(route, upstream, time.monotonic(), type(exc).__name__)
            raise UpstreamError() from None
        finally:
            await resp.aclose()

    def _wrap_usage(
        self, on_usage: UsageCallback | None, upstream: UpstreamConfig, req: ProxyRequest
    ) -> UsageCallback | None:
        if on_usage is None:
            return None
        llm = self._adapters[UpstreamKind.LLM]
        assert isinstance(llm, LlmAdapter)
        body = req.body if isinstance(req.body, bytes | bytearray) else None
        req_model = llm.request_model(bytes(body) if body else None)

        async def cb(usage: UsageRecord) -> None:
            await on_usage(replace(usage, provider=upstream.provider, model=usage.model or req_model))

        return cb

    def validate_credentials(self) -> None:
        """Startup check: every upstream.credential.env resolves. Raises ConfigError."""
        for upstream in self._policy.upstreams.values():
            if upstream.credential is not None:
                upstream.credential.resolve()
