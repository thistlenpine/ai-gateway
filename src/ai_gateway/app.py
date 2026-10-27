"""FastAPI application factory. OWNER: core."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_gateway.config import load_config
from ai_gateway.context import REQUEST_ID_HEADER, new_request_id, reset_ctx, set_ctx
from ai_gateway.errors import BadRequestError, GatewayError
from ai_gateway.models import RequestContext
from ai_gateway.observability import instrument_app, setup_telemetry
from ai_gateway.observability.logging import current_trace_ids
from ai_gateway.observability.otel import instrument_httpx_client
from ai_gateway.pipeline import GatewayPipeline, error_response
from ai_gateway.services import Services, build_services

log = logging.getLogger(__name__)

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


class RequestContextMiddleware:
    """Pure ASGI (NOT BaseHTTPMiddleware: it breaks streaming). Stage 0 of the pipeline:
    accepts/mints the request id, binds RequestContext to the ContextVar, echoes X-Request-ID."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        inbound = headers.get(b"x-request-id", b"").decode("latin-1") or None
        request_id = new_request_id(inbound)
        raw = scope.get("raw_path")
        path = bytes(raw).split(b"?", 1)[0].decode("latin-1") if raw else scope.get("path", "")
        client = scope.get("client")
        ctx = RequestContext(
            request_id=request_id,
            method=scope.get("method", ""),
            path=path,
            client_ip=client[0] if client else "",
            trace_id=current_trace_ids()[0],
        )
        state: MutableMapping[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                hdrs = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                hdrs.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode("latin-1")))
                message = {**message, "headers": hdrs}
            await send(message)

        token = set_ctx(ctx)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            reset_ctx(token)


class RequestTargetGuard:
    """Outermost pure-ASGI guard: any non-ASCII byte in the raw request target or query string is a
    400 before instrumentation/authn (SEC-11/12). Clients percent-encode; raw 8-bit bytes would be
    re-read as Latin-1 here, as UTF-8 by the OTel wrapper and as who-knows-what by the upstream."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            raw = bytes(scope.get("raw_path") or b"")
            query = bytes(scope.get("query_string") or b"")
            if not (raw.isascii() and query.isascii()):
                headers = {k.lower(): v for k, v in scope.get("headers", [])}
                rid = new_request_id(headers.get(b"x-request-id", b"").decode("latin-1") or None)
                resp = error_response(BadRequestError("Invalid request target"), rid)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(config_path: str | None = None, *, services: Services | None = None) -> FastAPI:
    """config_path defaults to env GATEWAY_CONFIG (default ./config/gateway.yaml).

    - load_config() -> build_services() (or use injected `services`, for tests)
    - lifespan: validator.warm() on startup; services.aclose() on shutdown
    - routes: GET /healthz, GET /readyz, and one catch-all `/{path:path}`
    - exception handlers: GatewayError -> JSON error body (+ headers); Exception -> 500
    - observability.instrument_app(app, cfg.telemetry) BEFORE the http clients are built
    - pure-ASGI middleware for request-id / RequestContext
    """
    if services is not None:
        cfg = services.config
        telemetry = services.telemetry
    else:
        cfg = load_config(config_path or os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml"))
        telemetry = setup_telemetry(cfg.gateway.telemetry)

    holder: dict[str, Services] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        svc = holder["services"]
        try:
            await svc.validator.warm()
        except Exception:  # IdP down at boot must not kill the process; /readyz reports it
            log.warning("JWKS warm-up failed", extra={"event": "startup.jwks_warm_failed"})
        try:
            yield
        finally:
            await svc.aclose()

    app = FastAPI(
        title="ai-gateway",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(RequestContextMiddleware)
    instrument_app(app, cfg.gateway.telemetry)  # must precede client construction
    # Wrap the whole built stack so the guard sits outside the OTel ASGI wrapper (which decodes the
    # query string and would raise on 8-bit bytes before any of our code runs).
    build_stack = app.build_middleware_stack
    app.build_middleware_stack = lambda: RequestTargetGuard(build_stack())  # type: ignore[method-assign]
    if services is None:
        svc = build_services(cfg, telemetry=telemetry)
    else:
        svc = services  # clients already exist: the global patch above does not reach them
        if cfg.gateway.telemetry.enabled:
            instrument_httpx_client(svc.upstream_http)
            instrument_httpx_client(svc.jwks_http)
    holder["services"] = svc
    app.state.services = svc
    pipeline = GatewayPipeline(svc)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> Response:
        return await pipeline.healthz()

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> Response:
        return await pipeline.readyz()

    @app.api_route("/{path:path}", methods=_METHODS, include_in_schema=False, response_model=None)
    async def catch_all(request: Request) -> Response:
        return await pipeline.handle(request)

    _install_error_handlers(app)
    return app


def _request_id(request: Request) -> str:
    return str(request.scope.get("state", {}).get("request_id") or new_request_id(None))


def _install_error_handlers(app: FastAPI) -> None:
    async def gateway_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, GatewayError)
        return error_response(exc, _request_id(request))

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        if exc.status_code == 405:
            err: GatewayError = BadRequestError("Method not allowed", code="method_not_allowed")
            err.status_code = 405
        else:
            err = GatewayError("Request failed", code="http_error")
            err.status_code = exc.status_code
        return error_response(err, _request_id(request))

    async def unhandled(request: Request, exc: Exception) -> Response:
        log.error("unhandled error", exc_info=exc, extra={"event": "request.error"})
        return error_response(GatewayError(), _request_id(request))

    handlers: dict[Any, Callable[[Request, Exception], Any]] = {
        GatewayError: gateway_error,
        StarletteHTTPException: http_error,
        Exception: unhandled,
    }
    for exc_type, handler in handlers.items():
        app.add_exception_handler(exc_type, handler)
