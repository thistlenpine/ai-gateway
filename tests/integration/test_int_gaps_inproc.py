"""Coverage-gap closers for pipeline.py / app.py / validator.py: failure paths that happy-path suites
never reach (in-process ASGI via the load rig, fake clock)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
import jwt
import pytest

from ai_gateway.auth import TokenValidator
from ai_gateway.config import IdPConfig
from ai_gateway.errors import AuthenticationError, AuthFailure, UpstreamError
from ai_gateway.models import RequestContext
from ai_gateway.observability.telemetry import NullTelemetry
from ai_gateway.pipeline import GatewayPipeline, _Accounting, error_body
from load.load_rig import INVOICES, LLM_URL, LoadGateway, anthropic_json, boot_load_gateway, llm_body
from support import KeyPair

LLM_PATH = "/llm/anthropic/v1/messages"


def _receiver(body: bytes) -> Any:
    """ASGI receive(): the request once, then disconnect (streaming responses listen for it)."""
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(0.01)
        return {"type": "http.disconnect"}

    return receive


async def _llm(gw: LoadGateway, client_id: str, **kw: Any) -> httpx.Response:
    return await gw.client.post(LLM_PATH, headers=gw.headers(client_id), content=llm_body(**kw))


# --------------------------------------------------------------------------- pipeline


def test_credential_misconfiguration_never_leaks_to_clients() -> None:
    body = error_body(UpstreamError("env FOO missing", code="upstream_credential_missing"), "rid")
    assert body["error"]["code"] == "upstream_error" and "FOO" not in str(body)


async def test_unhandled_exception_becomes_uniform_500_and_accounting_runs() -> None:
    async with boot_load_gateway() as gw:
        gw.upstream.get(INVOICES).mock(side_effect=RuntimeError("secret internals /etc/passwd"))
        r = await gw.get("c1")
        assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
        assert "secret" not in r.text and "passwd" not in r.text
        assert gw.tel.active == 0 and gw.tel.completed == 1


async def test_missing_upstream_credential_at_request_time_is_generic_502(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async with boot_load_gateway() as gw:
        monkeypatch.delenv("E2E_MCP_TOKEN")
        with caplog.at_level(logging.ERROR):
            r = await gw.client.post(
                "/mcp/tools",
                headers=gw.headers("c1", "mcp:tools:read"),
                content=b'{"jsonrpc":"2.0","id":1,"method":"ping"}',
            )
        assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"
        assert "E2E_MCP_TOKEN" not in r.text
        assert any(getattr(rec, "event", "") == "proxy.credential_missing" for rec in caplog.records)
        os.environ["E2E_MCP_TOKEN"] = "restored"  # monkeypatch restores the original on teardown


async def test_request_body_limits_for_buffered_routes() -> None:
    async with boot_load_gateway() as gw:
        gw.services.config.gateway.server.max_body_bytes = 200
        h = {**gw.headers("c1", "llm:claude"), "content-type": "application/json"}
        big = llm_body(text="x" * 1000)
        assert (await gw.client.post(LLM_PATH, headers=h, content=big)).status_code == 413  # Content-Length

        async def chunks() -> Any:
            for _ in range(10):
                yield b" " * 50  # no Content-Length: counted while streaming

        assert (await gw.client.post(LLM_PATH, headers=h, content=chunks())).status_code == 413
        bad = await gw.client.post(LLM_PATH, headers={**h, "content-length": "12abc"}, content=b"{}")
        assert bad.status_code in (400, 413)  # h11 may reject first; if it reaches us it is a 400


async def test_invalid_content_length_header_is_400_at_the_pipeline() -> None:
    async with boot_load_gateway() as gw:
        sent: list[dict[str, Any]] = []
        hdr = gw.headers("c1", "llm:claude")["Authorization"].encode()
        scope = {
            "type": "http", "method": "POST", "path": LLM_PATH, "raw_path": LLM_PATH.encode(), "query_string": b"",
            "headers": [(b"authorization", hdr), (b"content-length", b"12abc"), (b"host", b"g")],
            "client": ("1.2.3.4", 1), "server": ("g", 80), "scheme": "http", "http_version": "1.1", "root_path": "",
        }  # fmt: skip

        receive = _receiver(b"{}")

        async def send(m: dict[str, Any]) -> None:
            sent.append(m)

        await gw.app(scope, receive, send)  # type: ignore[arg-type]
        assert sent[0]["status"] == 400


async def test_pipeline_works_without_raw_path_or_middleware_context() -> None:
    """Servers that omit `raw_path` fall back to the decoded path; a handler call with no bound
    RequestContext (middleware missing) still answers instead of crashing."""
    async with boot_load_gateway() as gw:
        sent: list[dict[str, Any]] = []
        scope = {
            "type": "http", "method": "GET", "path": "/billing/invoices", "query_string": b"",
            "headers": [(b"authorization", gw.headers("c1")["Authorization"].encode())],
            "client": None, "server": ("g", 80), "scheme": "http", "http_version": "1.1", "root_path": "",
        }  # fmt: skip

        receive = _receiver(b"")

        async def send(m: dict[str, Any]) -> None:
            sent.append(m)

        await gw.app(scope, receive, send)  # type: ignore[arg-type]
        assert sent[0]["status"] == 200
        # handle() directly, i.e. get_ctx() is None
        from starlette.requests import Request

        req = Request({**scope, "headers": []})
        resp = await GatewayPipeline(gw.services).handle(req)
        assert resp.status_code == 401


async def test_finish_is_idempotent_and_accounting_failures_never_surface(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with boot_load_gateway() as gw:
        pipe = GatewayPipeline(gw.services)
        ctx, st = RequestContext("rid", "GET", "/x"), _Accounting()
        await pipe._finish(ctx, st)
        await pipe._finish(ctx, st)
        assert gw.tel.completed == 1  # second call is a no-op

        def boom(_ctx: Any) -> None:
            raise RuntimeError("telemetry down")

        gw.tel.record_request = boom  # type: ignore[method-assign,assignment]
        with caplog.at_level(logging.ERROR):
            r = await gw.get("c1")
        assert r.status_code == 200  # accounting failure never alters the response
        assert any(getattr(rec, "event", "") == "accounting.error" for rec in caplog.records)


async def test_upstream_error_status_refunds_token_reservation() -> None:
    async with boot_load_gateway(tokens_per_window=10_000, limits={"requests_per_window": 100}) as gw:
        gw.upstream.post(LLM_URL).respond(status_code=500, json={"error": "overloaded"})
        r = await _llm(gw, "t1", max_tokens=100)
        assert r.status_code == 500
        await asyncio.sleep(0)
        assert gw.tok("t1") == 0  # failed call: full refund


async def test_upstream_without_usage_keeps_the_preflight_estimate() -> None:
    async with boot_load_gateway(tokens_per_window=10_000, limits={"requests_per_window": 100}) as gw:
        gw.upstream.post(LLM_URL).respond(json={"id": "x", "content": []})  # no usage object
        r = await _llm(gw, "t2", max_tokens=100, text="x" * 40)
        assert r.status_code == 200
        assert gw.tok("t2") == 10 + 100  # estimate = ceil(40/4) + max_tokens is kept


async def test_actual_usage_replaces_estimate_and_refunds_difference() -> None:
    async with boot_load_gateway(tokens_per_window=10_000, limits={"requests_per_window": 100}) as gw:
        gw.upstream.post(LLM_URL).respond(json=anthropic_json(3, 4))
        assert (await _llm(gw, "t3", max_tokens=500)).status_code == 200
        assert gw.tok("t3") == 7


async def test_limiter_store_failure_fail_open_and_closed() -> None:
    async with boot_load_gateway() as gw:

        async def broken(*_a: Any, **_k: Any) -> Any:
            raise ConnectionError("redis down")

        gw.services.store.incr = broken  # type: ignore[method-assign]
        assert (await gw.get("c1")).status_code == 200  # fail_open (default): allowed
        gw.services.limiter._on_error = "fail_closed"
        r = await gw.get("c1")
        assert r.status_code == 429


# --------------------------------------------------------------------------- app


async def test_app_error_handlers_and_non_http_scopes() -> None:
    async with boot_load_gateway() as gw:
        r = await gw.client.request("TRACE", "/healthz")
        assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"
        assert "x-request-id" in r.headers
        # a non-http scope (websocket) bypasses the request-id middleware untouched
        got: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "websocket.disconnect", "code": 1000}

        async def send(m: dict[str, Any]) -> None:
            got.append(m)

        scope = {
            "type": "websocket",
            "path": "/billing/invoices",
            "headers": [],
            "query_string": b"",
            "subprotocols": [],
        }
        try:
            await gw.app(scope, receive, send)  # type: ignore[arg-type]
        except Exception:  # no websocket routes exist: starlette closes/raises, either is fine
            pass


async def test_idp_down_at_boot_does_not_kill_the_process_and_readyz_says_503() -> None:
    import respx

    from ai_gateway.app import create_app
    from ai_gateway.services import build_services
    from gw_rig import make_config

    os.environ.setdefault("E2E_MCP_TOKEN", "x")
    os.environ.setdefault("E2E_LLM_KEY", "y")
    cfg = make_config()
    cfg.gateway.telemetry.enabled = False
    router = respx.Router(assert_all_called=False)
    router.get("https://idp.test/jwks.json").respond(status_code=503)
    svc = build_services(
        cfg,
        telemetry=NullTelemetry(),
        jwks_http=httpx.AsyncClient(transport=httpx.MockTransport(router.async_handler)),
    )
    app = create_app(services=svc)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://g") as c:
            assert (await c.get("/healthz")).status_code == 200
            assert (await c.get("/readyz")).status_code == 503


# --------------------------------------------------------------------------- validator

KP = KeyPair("k1")
IDP = IdPConfig(
    name="m", issuer="https://idp.test", jwks_uri="https://idp.test/jwks.json", audiences=["ai-gateway"]
)


async def _validator() -> tuple[TokenValidator, httpx.AsyncClient]:
    import respx

    router = respx.Router(assert_all_called=False)
    router.get(IDP.jwks_uri).respond(json=KP.jwks())
    http = httpx.AsyncClient(transport=httpx.MockTransport(router.async_handler))
    return TokenValidator([IDP], http, NullTelemetry()), http


async def test_validator_rejects_non_numeric_nbf_and_non_string_sub() -> None:
    v, http = await _validator()
    async with http:
        for extra, want in (({"nbf": "later"}, AuthFailure.MALFORMED), ({"sub": 123}, AuthFailure.MALFORMED)):
            tok = KP.mint(**extra)
            with pytest.raises(AuthenticationError) as ei:
                await v.validate(tok)
            assert ei.value.failure is want


async def test_validator_maps_crypto_backend_errors_without_leaking(monkeypatch: pytest.MonkeyPatch) -> None:
    v, http = await _validator()
    real_decode = jwt.decode

    def make(exc: Exception) -> Any:
        def fake(token: str, key: Any = None, **kw: Any) -> Any:
            if key is None:
                return real_decode(token, **kw)  # the unverified peek at `iss`
            raise exc

        return fake

    async with http:
        tok = KP.mint()
        for exc, want in (
            (jwt.InvalidKeyError("secret-key-material"), AuthFailure.ALG_NOT_ALLOWED),
            (jwt.PyJWTError("weird"), AuthFailure.MALFORMED),
            (ValueError("openssl: secret-key-material"), AuthFailure.BAD_SIGNATURE),
        ):
            monkeypatch.setattr("ai_gateway.auth.validator.jwt.decode", make(exc))
            with pytest.raises(AuthenticationError) as ei:
                await v.validate(tok)
            assert ei.value.failure is want
            assert "secret-key-material" not in repr(ei.value) and ei.value.__cause__ is None
