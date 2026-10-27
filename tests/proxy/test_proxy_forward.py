from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from proxy_helpers import (
    BASE,
    AsyncStream,
    RecTelemetry,
    chunks,
    collect,
    make_proxy,
    request,
    route,
    upstream_cfg,
)

from ai_gateway.config import ConfigError, DefaultsConfig, PolicyConfig, UpstreamCredential
from ai_gateway.errors import (
    BadRequestError,
    PayloadTooLargeError,
    UpstreamError,
    UpstreamTimeoutError,
)
from ai_gateway.models import UpstreamKind
from ai_gateway.observability.telemetry import NullTelemetry
from ai_gateway.proxy import Proxy, build_http_client


def seen_headers(req: httpx.Request) -> dict[str, str]:
    return {k.lower(): v for k, v in req.headers.items()}


# ------------------------------------------------------------------ headers


async def test_header_hygiene(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UP_KEY", "s3cret")
    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200, content=b"ok", headers={"Connection": "close", "X-Up": "1"})

    cfg = upstream_cfg(credential=UpstreamCredential(env="UP_KEY"))
    proxy = make_proxy(handler, cfg)
    inbound = [
        ("Authorization", "Bearer CALLER-TOKEN"),
        ("Cookie", "sid=1"),
        ("X-Forwarded-For", "6.6.6.6"),
        ("X-Forwarded-Host", "evil"),
        ("X-Real-IP", "6.6.6.6"),
        ("Forwarded", "for=6.6.6.6"),
        ("X-Gateway-Client-Id", "admin"),
        ("X-Gateway-Subject", "root"),
        ("X-Request-ID", "forged"),
        ("Connection", "X-Hop, keep-alive"),
        ("X-Hop", "1"),
        ("Keep-Alive", "timeout=5"),
        ("Transfer-Encoding", "chunked"),
        ("Upgrade", "websocket"),
        ("Host", "evil.example"),
        ("traceparent", "00-" + "a" * 32 + "-" + "b" * 16 + "-01"),
        ("Mcp-Session-Id", "sess"),
        ("Accept", "application/json"),
        ("X-Custom", "keep"),
    ]
    resp = await proxy.forward(route(), request(headers=inbound))
    await collect(resp)
    h = seen_headers(captured[0])
    assert h["authorization"] == "Bearer s3cret"
    assert "CALLER-TOKEN" not in str(captured[0].headers)
    for gone in ("cookie", "x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded", "x-hop"):
        assert gone not in h
    assert "upgrade" not in h and "keep-alive" not in h
    assert h["host"] == "up.internal:8443"
    assert h["x-request-id"] == "req-12345678"
    assert h["x-gateway-client-id"] == "client-a"
    assert h["x-gateway-subject"] == "sub-1"
    assert h["mcp-session-id"] == "sess" and h["x-custom"] == "keep" and h["accept"] == "application/json"
    assert h.get("traceparent") != "00-" + "a" * 32 + "-" + "b" * 16 + "-01"  # no client trace forgery
    rh = {k.lower() for k, _ in resp.headers}
    assert "connection" not in rh and "x-up" in rh


async def test_caller_authorization_never_forwarded_without_credential() -> None:
    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200)

    proxy = make_proxy(handler)
    await collect(
        await proxy.forward(route(), request(headers=[("authorization", "Bearer T"), ("x-api-key", "k")]))
    )
    assert "authorization" not in captured[0].headers and "x-api-key" not in captured[0].headers


async def test_raw_credential_replaces_same_named_inbound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UP_KEY", "k-up")
    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200)

    cfg = upstream_cfg(credential=UpstreamCredential(env="UP_KEY", header="x-api-key", scheme=None))
    await collect(await make_proxy(handler, cfg).forward(route(), request(headers=[("X-API-Key", "caller")])))
    assert captured[0].headers.get_list("x-api-key") == ["k-up"]


async def test_missing_credential_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UP_KEY", raising=False)
    called = False

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    proxy = make_proxy(handler, upstream_cfg(credential=UpstreamCredential(env="UP_KEY")))
    with pytest.raises(UpstreamError) as ei:
        await proxy.forward(route(), request())
    assert ei.value.code == "upstream_credential_missing" and not called
    with pytest.raises(ConfigError):
        proxy.validate_credentials()


async def test_identity_headers_opt_out() -> None:
    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200)

    proxy = make_proxy(handler, upstream_cfg(forward_identity_headers=False))
    await collect(await proxy.forward(route(), request()))
    assert "x-gateway-client-id" not in captured[0].headers
    assert captured[0].headers["x-request-id"] == "req-12345678"


async def test_traceparent_injected_from_active_span() -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200)

    tracer = TracerProvider().get_tracer("t")
    with tracer.start_as_current_span("srv") as span:
        proxy = make_proxy(handler)
        await collect(
            await proxy.forward(
                route(), request(headers=[("traceparent", "00-" + "1" * 32 + "-" + "2" * 16 + "-01")])
            )
        )
        tid = format(span.get_span_context().trace_id, "032x")
    assert trace.get_current_span() is not span
    assert captured[0].headers["traceparent"].split("-")[1] == tid


# ------------------------------------------------------------------ URL / SSRF


@pytest.mark.parametrize(
    ("rel", "strip", "expected"),
    [
        ("/items/1", True, "/api/items/1"),
        ("/items/1", False, "/api/svc/items/1"),
        ("/", True, "/api"),  # prefix root: no invented trailing slash (SEC-07b)
        ("/a b/ü", True, "/api/a%20b/%C3%BC"),
        ("/100%", True, "/api/100%25"),
        ("/a?b#c", True, "/api/a%3Fb%23c"),
    ],
)
async def test_url_join(rel: str, strip: bool, expected: str) -> None:
    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200)

    proxy = make_proxy(handler, upstream_cfg(strip_prefix=strip))
    await collect(await proxy.forward(route(rel=rel), request(path=rel, query="q=1&r=%2F")))
    u = captured[0].url
    assert (u.scheme, u.host, u.port) == ("https", "up.internal", 8443)
    assert u.raw_path.decode() == expected + "?q=1&r=%2F"


@pytest.mark.parametrize(
    "rel",
    [
        "//evil.com/x",
        "/../admin",
        "/a/../../b",
        "/a/./b",
        "/a\\b",
        "/a\x00b",
        "/a\r\nHost: evil",
        "http://evil.com/",
        "@evil.com/x",
        "evil",
    ],
)
async def test_path_attacks_rejected(rel: str) -> None:
    called = False

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    proxy = make_proxy(handler)
    with pytest.raises(BadRequestError):
        await proxy.forward(route(rel=rel), request(path=rel))
    assert not called


@pytest.mark.parametrize("q", ["a=1\r\nX: y", "a=1 b", "a=1#frag", "a=\x00", "a=\\"])
async def test_query_attacks_rejected(q: str) -> None:
    proxy = make_proxy(lambda r: httpx.Response(200))
    with pytest.raises(BadRequestError):
        await proxy.forward(route(), request(query=q))


async def test_host_header_and_absolute_target_cannot_redirect_destination() -> None:
    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200)

    proxy = make_proxy(handler)
    await collect(
        await proxy.forward(
            route(rel="/@evil.com:80/x"),
            request(headers=[("Host", "evil.com"), ("X-Forwarded-Host", "evil.com")]),
        )
    )
    assert captured[0].url.host == "up.internal"
    assert captured[0].headers["host"] == "up.internal:8443"
    assert captured[0].url.raw_path.startswith(b"/api/@evil.com:80/x")


@pytest.mark.parametrize("method", ["CONNECT", "TRACE", "GE T", "get\r\n", ""])
async def test_bad_methods(method: str) -> None:
    proxy = make_proxy(lambda r: httpx.Response(200))
    with pytest.raises(BadRequestError):
        await proxy.forward(route(), request(method=method))


async def test_redirects_not_followed() -> None:
    calls = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/"})

    proxy = Proxy(
        PolicyConfig(upstreams={"u": upstream_cfg()}),
        build_http_client(DefaultsConfig()),
        NullTelemetry(),
        DefaultsConfig(),
    )
    # Real client built by the factory must not follow redirects.
    assert proxy._http.follow_redirects is False
    proxy._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    resp = await proxy.forward(route(), request())
    await collect(resp)
    assert resp.status == 302 and calls == 1
    assert "location" not in dict(resp.headers)  # foreign/internal Location is never relayed (SEC-07)


def test_build_http_client() -> None:
    c = build_http_client(DefaultsConfig(upstream_timeout_s=7, upstream_connect_timeout_s=2))
    assert c.follow_redirects is False and c.trust_env is False
    assert c.timeout.connect == 2 and c.timeout.read == 7


# ------------------------------------------------------------------ errors / timeouts


@pytest.mark.parametrize(
    ("exc", "err", "code", "etype"),
    [
        (httpx.ReadTimeout("t"), UpstreamTimeoutError, 504, "timeout"),
        (httpx.ConnectTimeout("t"), UpstreamTimeoutError, 504, "timeout"),
        (httpx.ConnectError("boom"), UpstreamError, 502, "ConnectError"),
    ],
)
async def test_upstream_errors(exc: Exception, err: type[Exception], code: int, etype: str) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise exc

    tel = RecTelemetry()
    proxy = make_proxy(handler, telemetry=tel)
    with pytest.raises(err) as ei:
        await proxy.forward(route(), request())
    assert ei.value.status_code == code  # type: ignore[attr-defined]
    assert tel.upstream_calls[-1]["error_type"] == etype and tel.upstream_calls[-1]["status"] is None
    assert "boom" not in str(ei.value)


@respx.mock
async def test_respx_timeout_and_status_metrics() -> None:
    respx.get(f"{BASE}/slow").mock(side_effect=httpx.ReadTimeout("x"))
    respx.get(f"{BASE}/err").mock(return_value=httpx.Response(503, text="down"))
    tel = RecTelemetry()
    proxy = Proxy(
        PolicyConfig(upstreams={"u": upstream_cfg()}),
        httpx.AsyncClient(),
        tel,
        DefaultsConfig(),
    )
    with pytest.raises(UpstreamTimeoutError):
        await proxy.forward(route(rel="/slow"), request(path="/slow"))
    resp = await proxy.forward(route(rel="/err"), request(path="/err"))
    assert resp.status == 503 and await collect(resp) == b"down"
    assert tel.upstream_calls[-1]["status"] == 503


async def test_per_upstream_timeout_applied() -> None:
    seen: list[dict[str, object]] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r.extensions["timeout"])
        return httpx.Response(200)

    proxy = make_proxy(handler, upstream_cfg(timeout_s=3.5))
    await collect(await proxy.forward(route(), request()))
    assert seen[0]["read"] == 3.5 and seen[0]["connect"] == 5.0


# ------------------------------------------------------------------ bodies / streaming / caps


async def test_request_body_streamed_with_content_length() -> None:
    captured: list[httpx.Request] = []

    async def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200, content=await r.aread())

    proxy = make_proxy(handler)
    resp = await proxy.forward(
        route(),
        request(
            "POST",
            body=chunks(b"ab", b"cd"),
            headers=[("Content-Length", "4"), ("Transfer-Encoding", "chunked")],
        ),
    )
    assert await collect(resp) == b"abcd"
    assert captured[0].headers["content-length"] == "4" and "transfer-encoding" not in captured[0].headers


async def test_request_size_caps() -> None:
    proxy = make_proxy(lambda r: httpx.Response(200), max_request_bytes=10)
    with pytest.raises(PayloadTooLargeError):
        await proxy.forward(route(), request("POST", body=b"x" * 11))
    with pytest.raises(PayloadTooLargeError):
        await proxy.forward(route(), request("POST", body=chunks(b"x"), headers=[("Content-Length", "99")]))
    with pytest.raises(BadRequestError):
        await proxy.forward(route(), request("POST", body=chunks(b"x"), headers=[("Content-Length", "-1")]))

    async def handler(r: httpx.Request) -> httpx.Response:
        await r.aread()
        return httpx.Response(200)

    proxy = make_proxy(handler, max_request_bytes=10)
    with pytest.raises(PayloadTooLargeError):  # lying / absent Content-Length, over cap mid-stream
        await proxy.forward(route(), request("POST", body=chunks(b"x" * 6, b"y" * 6)))


async def test_response_size_cap() -> None:
    proxy = make_proxy(lambda r: httpx.Response(200, content=b"x" * 100), max_response_bytes=50)
    with pytest.raises(UpstreamError):
        await proxy.forward(route(), request())

    def chunked(r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=AsyncStream([b"x" * 30, b"y" * 30]))

    proxy = make_proxy(chunked, max_response_bytes=50)
    resp = await proxy.forward(route(), request())
    with pytest.raises(UpstreamError):
        await collect(resp)


async def test_sse_relayed_incrementally() -> None:
    # (httpx.ASGITransport buffers whole responses, so gate the upstream stream by hand.)
    gate = asyncio.Event()

    class Gated(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"data: 1\n\n"
            await gate.wait()
            yield b"data: 2\n\n"

    proxy = make_proxy(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Gated())
    )
    resp = await proxy.forward(route(), request(headers=[("Accept", "text/event-stream")]))
    assert resp.is_stream and resp.status == 200
    it = resp.body.__aiter__()
    assert await asyncio.wait_for(it.__anext__(), 2) == b"data: 1\n\n"  # before upstream finished
    gate.set()
    assert await asyncio.wait_for(it.__anext__(), 2) == b"data: 2\n\n"
    with pytest.raises(StopAsyncIteration):
        await it.__anext__()
    await resp.aclose()


async def test_aclose_closes_upstream_and_is_idempotent() -> None:
    closed = asyncio.Event()

    class S(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b"data: 1\n\n"
            await asyncio.sleep(60)

        async def aclose(self) -> None:
            closed.set()

    proxy = make_proxy(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=S())
    )
    resp = await proxy.forward(route(), request())
    it = resp.body.__aiter__()
    await it.__anext__()
    await resp.aclose()
    await resp.aclose()
    assert closed.is_set()


async def test_compression_passthrough_for_http_identity_for_llm() -> None:
    import gzip

    payload = gzip.compress(b"hello")
    seen: list[str | None] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r.headers.get("accept-encoding"))
        return httpx.Response(200, stream=AsyncStream([payload]), headers={"content-encoding": "gzip"})

    proxy = make_proxy(handler)
    resp = await proxy.forward(route(), request(headers=[("Accept-Encoding", "gzip")]))
    assert await collect(resp) == payload and ("content-encoding", "gzip") in [
        (k.lower(), v) for k, v in resp.headers
    ]
    assert seen[0] == "gzip"

    proxy = make_proxy(handler, upstream_cfg(UpstreamKind.LLM))
    resp = await proxy.forward(route(UpstreamKind.LLM), request(headers=[("Accept-Encoding", "gzip")]))
    assert await collect(resp) == b"hello"
    assert "content-encoding" not in {k.lower() for k, _ in resp.headers}
    assert seen[1] == "identity"


async def test_unknown_upstream_and_kind_mismatch() -> None:
    proxy = make_proxy(lambda r: httpx.Response(200))
    from ai_gateway.models import RouteMatch

    with pytest.raises(UpstreamError):
        await proxy.forward(RouteMatch("nope", UpstreamKind.HTTP, "/x", "/"), request())
    with pytest.raises(UpstreamError):
        await proxy.forward(route(UpstreamKind.MCP), request())
