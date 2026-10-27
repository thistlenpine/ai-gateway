"""Request pipeline orchestration. OWNER: core.

Order (each stage = one span named in observability.schema; any GatewayError short-circuits to
the error responder, which still records metrics + access log):

  0 request-id    done by app.RequestContextMiddleware: RequestContext bound to a ContextVar,
                  X-Request-ID echoed on every response. The pipeline does add_active(+1).
  0b ip limit     services.ip_limiter.check(ctx.client_ip) -> 429 `ip_rate_limited` (+Retry-After)
                  BEFORE any token work (cheap pre-auth flood backstop; per instance)
  1 authn         extract_bearer -> validator.validate -> ctx.principal, ctx.client_label
                  (unregistered/disabled client + defaults.unknown_clients=="deny" -> 403)
  1b block-check  limiter.is_blocked(client_id) -> 429 `client_blocked` BEFORE reading the body
  2 authz         route = policy.resolve_route(normalized path) (None -> deny NO_ROUTE);
                  buffer body for mcp/llm (<= server.max_body_bytes, else 413);
                  adapter.classify -> resources; policy.authorize_all -> PolicyDecision; audit
                  event for allow AND deny; deny -> 403 `access_denied` (uniform body)
  3 rate limit    limiter.check_request: DELAY -> apply_delay; REJECT/BLOCK -> 429.
                  LLM: reserve_tokens(estimate) -> REJECT -> 429 `token_quota_exceeded`.
  4 proxy         proxy.forward(...) -> streaming response; disconnect -> resp.aclose()
  5 accounting    runs exactly once when the body finishes/aborts (or the request fails):
                  release delay slot, reconcile/refund tokens, record_request, access log.

Unhandled exceptions -> 500 {"error": {"code": "internal_error"}}; stack trace to logs only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, NoReturn

import anyio
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ai_gateway.auth import extract_bearer
from ai_gateway.config import LimitPolicy
from ai_gateway.context import REQUEST_ID_HEADER, get_ctx, new_request_id
from ai_gateway.errors import (
    AuthenticationError,
    AuthFailure,
    AuthorizationError,
    BadRequestError,
    GatewayError,
    PayloadTooLargeError,
    RateLimitedError,
    UpstreamError,
)
from ai_gateway.models import (
    DecisionReason,
    LimitDecision,
    LimitScope,
    Phase,
    PolicyDecision,
    Principal,
    RequestContext,
    RouteMatch,
    UpstreamKind,
    UsageRecord,
)
from ai_gateway.observability import AuditEvent, AuditType, schema
from ai_gateway.observability.audit import authz_event
from ai_gateway.policy.matchers import normalize_path
from ai_gateway.proxy import ProxyRequest, ProxyResponse
from ai_gateway.services import Services

log = logging.getLogger(__name__)

# Codes that must never leak configuration details to the client.
_GENERIC_UPSTREAM = ("upstream_error", "Upstream request failed")


def error_body(err: GatewayError, request_id: str) -> dict[str, Any]:
    code, message = err.code, err.message
    if isinstance(err, UpstreamError) and err.status_code == 502 and code == "upstream_credential_missing":
        code, message = _GENERIC_UPSTREAM
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def error_response(err: GatewayError, request_id: str) -> JSONResponse:
    headers = dict(err.headers)
    headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(error_body(err, request_id), status_code=err.status_code, headers=headers)


@dataclass(slots=True)
class _Accounting:
    """Per-request mutable state that the finalizer needs (stage 5)."""

    delay: LimitDecision | None = None  # delay slot held and not yet released
    limits: LimitPolicy | None = None
    reserved: int = 0  # tokens reserved (LLM) and not yet reconciled
    streaming: bool = False  # the response body owns finalization
    finished: bool = False
    bytes_out: int = 0
    rl_phase: str = "allow"
    keep_estimate: bool = False  # upstream completed without usage: keep the pre-flight estimate
    conc: str = ""  # in-flight concurrency slot key held and not yet released


class GatewayPipeline:
    def __init__(self, services: Services) -> None:
        self._s = services

    # ------------------------------------------------------------------ health

    async def healthz(self) -> Response:
        """Liveness: 200 always (process up)."""
        return JSONResponse({"status": "ok"})

    async def readyz(self) -> Response:
        """Readiness: 200 iff validator.ready (JWKS cached) else 503. No auth, no spans."""
        if self._s.validator.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not_ready"}, status_code=503)

    # ------------------------------------------------------------------ entry

    async def handle(self, request: Request) -> Response:
        """Catch-all handler for every non-internal path."""
        ctx = get_ctx()
        if ctx is None:  # middleware missing (should not happen); degrade gracefully
            ctx = RequestContext(new_request_id(None), request.method, request.url.path)
        st = _Accounting()
        tel = self._s.telemetry
        tel.add_active(1)
        try:
            resp = await self._run(request, ctx, st)
        except GatewayError as err:
            ctx.error_code = err.code
            resp = error_response(err, ctx.request_id)
        except Exception:
            log.exception("unhandled error", extra={"event": "request.error"})
            ctx.error_code = "internal_error"
            resp = error_response(GatewayError(), ctx.request_id)
        except BaseException:  # cancellation / client disconnect before a response existed
            ctx.error_code = ctx.error_code or "client_disconnected"
            ctx.status_code = ctx.status_code or 499
            await self._finish(ctx, st)
            raise
        ctx.status_code = resp.status_code
        if not st.streaming:
            st.bytes_out = len(resp.body)
            await self._finish(ctx, st)
        return resp

    # ------------------------------------------------------------------ stages

    async def _run(self, request: Request, ctx: RequestContext, st: _Accounting) -> Response:
        s = self._s
        self._ip_limit(ctx)  # 0: cheap pre-auth limiter, before any token work
        principal = await self._authenticate(request, ctx)
        client_id = ctx.limit_key  # namespaced by IdP (+ subject for end-user tokens): SEC-08
        limits = s.config.limits_for(principal.client_id, principal.issuer)
        st.limits = limits

        # 1b: block fast path, before any body read / policy work.
        with s.telemetry.span(schema.SPAN_RATELIMIT, {schema.A_CLIENT_ID: ctx.client_label}):
            blocked = await s.limiter.is_blocked(client_id)
        if blocked is not None:
            ctx.limit = blocked
            st.rl_phase = Phase.BLOCK.value
            raise RateLimitedError(
                "Client temporarily blocked", code="client_blocked", headers=blocked.headers()
            )

        try:
            route, decision, body = await self._authorize(request, ctx, principal)
        except (AuthorizationError, BadRequestError, PayloadTooLargeError) as err:
            # Denied/malformed requests are not free: they count against the client's window (SEC-03).
            await self._charge_denial(ctx, limits, err)
            raise

        ctx.policy = decision  # before rate limiting: the token estimate uses the grant's clamp
        # 3: rate limiting
        await self._rate_limit(ctx, st, limits, route.kind, body)

        # 3b: bound concurrently in-flight proxied requests/streams per client (SEC-22)
        slot = await s.limiter.acquire_concurrency(client_id, limits)
        if slot is None:
            raise RateLimitedError(
                "Too many concurrent requests",
                code="too_many_concurrent_requests",
                headers={"Retry-After": "1"},
            )
        st.conc = slot

        # 4: proxy
        stream: Any = body
        if route.kind is UpstreamKind.HTTP and _has_body(request):
            stream = request.stream()  # bodiless requests must not become chunked uploads
        preq = ProxyRequest(
            method=request.method,
            path=route.relative_path,
            query=request.scope.get("query_string", b"").decode("latin-1"),
            headers=[(k.decode("latin-1"), v.decode("latin-1")) for k, v in request.headers.raw],
            body=stream,
            ctx=ctx,
        )
        tool_allowed = (
            s.policy.tool_filter(principal, route.upstream) if route.kind is UpstreamKind.MCP else None
        )

        async def on_usage(usage: UsageRecord) -> None:
            await self._on_usage(ctx, st, usage)

        try:
            presp = await s.proxy.forward(
                route,
                preq,
                tool_allowed=tool_allowed,
                on_usage=on_usage if route.kind is UpstreamKind.LLM else None,
            )
        except UpstreamError as err:
            if err.code == "upstream_credential_missing":
                log.error("upstream credential missing", extra={"event": "proxy.credential_missing"})
            raise
        try:
            return self._build_response(ctx, st, presp)
        except BaseException:
            await presp.aclose()
            raise

    def _ip_limit(self, ctx: RequestContext) -> None:
        wait = self._s.ip_limiter.check(ctx.client_ip)
        if wait is None:
            return
        cfg = self._s.config.gateway.server.ip_rate_limit
        decision = LimitDecision(
            phase=Phase.REJECT,
            scope=LimitScope.IP,
            client_id="ip",  # never the address itself: bounded metric labels, no PII in metrics
            limit=cfg.requests_per_window,
            remaining=0,
            reset_s=wait,
            retry_after_s=wait,
            window_s=cfg.window_seconds,
            reason="ip_rate_limited",
        )
        self._s.telemetry.record_rate_limit(decision, ctx)
        raise RateLimitedError(
            "Rate limit exceeded",
            code="ip_rate_limited",
            headers={"Retry-After": self._s.ip_limiter.retry_after_header(wait)},
        )

    async def _charge_denial(self, ctx: RequestContext, limits: LimitPolicy, err: GatewayError) -> None:
        """Count a denied/malformed request against the limiter without ever sleeping."""
        decision = await self._s.limiter.check_request(ctx.limit_key, limits)
        if decision.phase is Phase.DELAY:
            await self._s.limiter.release(decision)  # counted; no delay for a request being refused
        elif not decision.allowed:
            ctx.limit = decision
            st_code = "client_blocked" if decision.phase is Phase.BLOCK else "rate_limited"
            raise RateLimitedError(
                "Client temporarily blocked" if st_code == "client_blocked" else "Rate limit exceeded",
                code=st_code,
                headers=decision.headers(),
            ) from err

    async def _authenticate(self, request: Request, ctx: RequestContext) -> Principal:
        s = self._s
        with s.telemetry.span(schema.SPAN_AUTHN):
            try:
                token = extract_bearer(request.headers.get("authorization"))
            except AuthenticationError as err:
                s.telemetry.record_auth_failure(err.failure.value)
                self._audit_authn_failure(ctx, err.failure)
                raise
            try:
                principal = await s.validator.validate(token)
            except AuthenticationError as err:
                self._audit_authn_failure(ctx, err.failure)
                raise
            ctx.principal = principal
            client = s.config.client_for(principal.client_id, principal.issuer)
            registered = client is not None
            ctx.client_label = principal.client_id if registered else "unregistered"
            ctx.limit_key = _limit_key(principal)
            disabled = client is not None and not client.enabled
            unknown = not registered and s.config.gateway.defaults.unknown_clients == "deny"
            if disabled or unknown:
                s.telemetry.record_auth_failure(
                    AuthFailure.CLIENT_NOT_REGISTERED.value, issuer=principal.issuer
                )
                self._audit_authn_failure(ctx, AuthFailure.CLIENT_NOT_REGISTERED)
                raise AuthorizationError()
            return principal

    def _audit_authn_failure(self, ctx: RequestContext, failure: AuthFailure) -> None:
        p = ctx.principal
        self._s.telemetry.audit(
            AuditEvent(
                type=AuditType.AUTHN_FAILURE,
                request_id=ctx.request_id,
                trace_id=ctx.trace_id,
                client_id=p.client_id if p else None,
                subject=p.subject if p else None,
                issuer=p.issuer if p else None,
                decision="deny",
                reason=failure.value,
                method=ctx.method,
                path=ctx.path,
                client_ip=ctx.client_ip or None,
            )
        )

    async def _authorize(
        self, request: Request, ctx: RequestContext, principal: Principal
    ) -> tuple[RouteMatch, PolicyDecision, bytes | None]:
        s = self._s
        with s.telemetry.span(schema.SPAN_AUTHZ):
            try:
                path = normalize_path(_raw_path(request))
            except ValueError:
                self._deny(ctx, PolicyDecision(False, DecisionReason.MALFORMED_REQUEST))
            ctx.path = path
            route = s.policy.resolve_route(path)
            if route is None:
                self._deny(ctx, PolicyDecision(False, DecisionReason.NO_ROUTE))
            ctx.route = route

            body: bytes | None = None
            if route.kind is not UpstreamKind.HTTP:
                # Scope-only check BEFORE touching the body: a caller with no grant for this upstream
                # gets the uniform 403 (no route/kind oracle, no buffering/parsing) - SEC-24.
                if not s.policy.has_any_grant(principal, route.upstream):
                    self._deny(ctx, PolicyDecision(False, DecisionReason.INSUFFICIENT_SCOPE))
                body = await self._read_body(request)
            adapter = s.proxy.adapter(route.kind)
            try:
                resources = adapter.classify(route, request.method, body)
            except BadRequestError:
                s.telemetry.audit(authz_event(ctx, PolicyDecision(False, DecisionReason.MALFORMED_REQUEST)))
                s.telemetry.record_authz(PolicyDecision(False, DecisionReason.MALFORMED_REQUEST), ctx)
                raise
            ctx.resources = list(resources)
            decision = s.policy.authorize_all(principal, resources)
            s.telemetry.audit(authz_event(ctx, decision))
            s.telemetry.record_authz(decision, ctx)
            if not decision.allowed:
                ctx.policy = decision
                raise AuthorizationError()
            return route, decision, body

    def _deny(self, ctx: RequestContext, decision: PolicyDecision) -> NoReturn:
        self._s.telemetry.audit(authz_event(ctx, decision))
        self._s.telemetry.record_authz(decision, ctx)
        raise AuthorizationError()

    async def _read_body(self, request: Request) -> bytes | None:
        limit = self._s.config.gateway.server.max_body_bytes
        declared = request.headers.get("content-length")
        if declared is not None:
            if not declared.isascii() or not declared.isdigit():
                raise BadRequestError("Invalid Content-Length")
            if int(declared) > limit:
                raise PayloadTooLargeError()
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > limit:
                raise PayloadTooLargeError()
        return bytes(buf) or None

    async def _rate_limit(
        self,
        ctx: RequestContext,
        st: _Accounting,
        limits: LimitPolicy,
        kind: UpstreamKind,
        body: bytes | None,
    ) -> None:
        s = self._s
        client_id = ctx.limit_key or "anonymous"
        with s.telemetry.span(schema.SPAN_RATELIMIT, {schema.A_CLIENT_ID: ctx.client_label}):
            decision = await s.limiter.check_request(client_id, limits)
            ctx.limit = decision
            st.rl_phase = decision.phase.value
            if decision.phase is Phase.DELAY:
                st.delay = decision  # finalizer releases the slot if apply_delay never runs
            if not decision.allowed:
                code = "client_blocked" if decision.phase is Phase.BLOCK else "rate_limited"
                raise RateLimitedError(
                    "Client temporarily blocked" if code == "client_blocked" else "Rate limit exceeded",
                    code=code,
                    headers=decision.headers(),
                )
            if decision.phase is Phase.DELAY:
                st.delay = None  # apply_delay releases the slot itself, exactly once
                with s.telemetry.span(schema.SPAN_RATELIMIT_DELAY):
                    await s.limiter.apply_delay(decision)

            if kind is UpstreamKind.LLM:
                cap = ctx.policy.max_tokens_per_request if ctx.policy else None
                estimate = s.proxy.adapter(kind).estimate_tokens(body, cap=cap)
                ctx.estimated_tokens = estimate
                tdec = await s.limiter.reserve_tokens(client_id, limits, estimate)
                ctx.token_limit = tdec
                if not tdec.allowed:
                    raise RateLimitedError(
                        "Token quota exceeded", code="token_quota_exceeded", headers=tdec.headers()
                    )
                if limits.tokens_per_window is not None:
                    st.reserved = estimate

    # ------------------------------------------------------------------ response + accounting

    def _build_response(self, ctx: RequestContext, st: _Accounting, presp: ProxyResponse) -> Response:
        # Everything that can fail runs BEFORE st.streaming is set (SEC-05): a header that cannot be
        # encoded is dropped, never allowed to wedge accounting or leak the upstream connection.
        raw: list[tuple[bytes, bytes]] = []
        for k, v in presp.headers:
            low = k.lower()
            if low == "x-request-id" or low.startswith("ratelimit-"):
                continue
            try:
                raw.append((low.encode("latin-1"), v.encode("latin-1")))
            except UnicodeEncodeError:
                continue
        extra = ctx.limit.headers() if ctx.limit is not None else {}
        raw.extend((k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in extra.items())

        ctx.status_code = presp.status
        st.streaming = True
        pipeline = self
        deadline = self._stream_deadline(ctx, st)

        async def body() -> AsyncIterator[bytes]:
            try:
                it = presp.body.__aiter__()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _StreamExpired
                    with anyio.move_on_after(remaining) as scope:
                        try:
                            chunk = await it.__anext__()
                        except StopAsyncIteration:
                            return
                    if scope.cancelled_caught:
                        raise _StreamExpired
                    st.bytes_out += len(chunk)
                    yield chunk
            except _StreamExpired:
                # max lifetime / token expiry: SSE ends cleanly (clients reconnect and re-authenticate);
                # any other body is aborted so a truncated payload never looks complete.
                ctx.error_code = ctx.error_code or "stream_expired"
                if not presp.is_stream:
                    raise
            except GatewayError as err:
                ctx.error_code = err.code
                raise
            except BaseException:
                ctx.error_code = ctx.error_code or "aborted"
                raise
            finally:
                with anyio.CancelScope(shield=True):
                    await presp.aclose()
                    await pipeline._finish(ctx, st)

        out = StreamingResponse(body(), status_code=presp.status)
        out.raw_headers = raw
        return out

    def _stream_deadline(self, ctx: RequestContext, st: _Accounting) -> float:
        """Monotonic deadline: min(max_stream_seconds, time until the token expires) (SEC-23)."""
        now = time.monotonic()
        life = st.limits.max_stream_seconds if st.limits is not None else 900.0
        deadline = now + life
        p = ctx.principal
        if p is not None and p.expires_at > 0:
            deadline = min(deadline, now + max(p.expires_at - time.time(), 0.0))
        return deadline

    async def _on_usage(self, ctx: RequestContext, st: _Accounting, usage: UsageRecord) -> None:
        s = self._s
        ctx.usage = usage
        failed = (ctx.status_code or 0) >= 400
        if not failed and not usage.estimated and usage.total > 0:
            s.telemetry.record_llm_tokens(
                usage, upstream=ctx.route.upstream if ctx.route else "", client_label=ctx.client_label
            )
        if st.reserved and st.limits is not None and ctx.principal is not None:
            if failed:
                actual = 0
            elif usage.estimated:
                # No usage observed. Whether to keep the pre-flight estimate (completed response) or
                # refund it (aborted stream) is only known once the body wrapper has finished: _finish().
                st.keep_estimate = True
                return
            else:
                actual = usage.total
            reserved, st.reserved = st.reserved, 0
            await s.limiter.reconcile_tokens(ctx.limit_key, st.limits, reserved, actual)

    async def _finish(self, ctx: RequestContext, st: _Accounting) -> None:
        """Stage 5. Idempotent; shielded from cancellation; never raises."""
        if st.finished:
            return
        st.finished = True
        s = self._s
        with anyio.CancelScope(shield=True):
            try:
                with s.telemetry.span(schema.SPAN_ACCOUNTING):
                    if st.delay is not None:
                        delay, st.delay = st.delay, None
                        await s.limiter.release(delay)
                    if st.conc:
                        conc, st.conc = st.conc, ""
                        await s.limiter.release_concurrency(conc)
                    if st.reserved and st.limits is not None and ctx.principal is not None:
                        reserved, st.reserved = st.reserved, 0
                        aborted = ctx.error_code in _ABORT_CODES
                        if aborted or not st.keep_estimate:  # no usable usage observed -> refund
                            await s.limiter.reconcile_tokens(ctx.limit_key, st.limits, reserved, 0)
                    s.telemetry.record_request(ctx)
                    s.telemetry.add_active(-1)
                    log.info(
                        "request completed",
                        extra={
                            "event": "request.completed",
                            "method": ctx.method,
                            "route": ctx.route.upstream if ctx.route else None,
                            "status": ctx.status_code,
                            "duration_ms": round(ctx.elapsed_s * 1000, 2),
                            "bytes": st.bytes_out,
                            "rl_phase": st.rl_phase,
                            "error_code": ctx.error_code,
                        },
                    )
            except Exception:
                log.exception("accounting failed", extra={"event": "accounting.error"})


_ABORT_CODES = frozenset({"aborted", "stream_expired", "client_disconnected"})


class _StreamExpired(Exception):
    """Internal: response outlived max_stream_seconds or the caller's token."""


def _limit_key(p: Principal) -> str:
    """Rate-limit identity (SEC-08): namespaced by IdP so equal client_ids from different issuers
    never share buckets/blocks; end-user tokens (sub != client_id) get their own bucket per user."""
    base = f"{p.idp or p.issuer}:{p.client_id}"
    return base if p.is_client_credentials else f"{base}|{p.subject}"


def _has_body(request: Request) -> bool:
    cl = request.headers.get("content-length")
    if cl is not None:
        return cl.strip() != "0"
    return "transfer-encoding" in request.headers


def _raw_path(request: Request) -> str:
    """Percent-encoded request path without the query (what normalize_path expects)."""
    raw = request.scope.get("raw_path")
    if isinstance(raw, bytes | bytearray) and raw:
        return bytes(raw).split(b"?", 1)[0].decode("latin-1")
    from urllib.parse import quote

    return quote(request.scope["path"], safe="/:@!$&'()*+,=-._~")


__all__ = ["GatewayPipeline", "error_body", "error_response"]
