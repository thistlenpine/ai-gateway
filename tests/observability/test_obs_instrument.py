from __future__ import annotations

import httpx
import respx
from fastapi import FastAPI, Request
from obs_rig import API_KEY, BEARER, Rig
from opentelemetry.trace import SpanKind

from ai_gateway.config import TelemetryConfig
from ai_gateway.observability.otel import instrument_app

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PARENT_ID = "b7ad6b7169203331"


async def test_server_span_proxy_child_client_span_and_traceparent(
    rig: Rig, respx_mock: respx.MockRouter
) -> None:
    seen: dict[str, str] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["traceparent"] = request.headers.get("traceparent", "")
        return httpx.Response(200, json={"ok": True})

    respx_mock.get("http://upstream/invoices").mock(side_effect=upstream)

    app = FastAPI()
    cfg = TelemetryConfig()
    instrument_app(app, cfg)  # before the client is created (httpx instrumentation patches new clients)
    client = httpx.AsyncClient(base_url="http://upstream")

    @app.get("/billing/{rest:path}")
    async def handler(request: Request, rest: str) -> dict[str, bool]:
        with rig.tel.span("gateway.proxy"):
            r = await client.get("/invoices", params={"api_key": API_KEY})
        return {"ok": r.status_code == 200}

    @app.get("/healthz")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.get(
            f"/billing/x?access_token={BEARER}&page=2",
            headers={"traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01", "authorization": f"Bearer {BEARER}"},
        )
        await c.get("/healthz")
    assert r.status_code == 200
    await client.aclose()

    spans = rig.spans.get_finished_spans()
    server = next(s for s in spans if s.kind is SpanKind.SERVER)
    proxy = next(s for s in spans if s.name == "gateway.proxy")
    client_span = next(s for s in spans if s.kind is SpanKind.CLIENT)
    # inbound W3C traceparent honoured
    assert f"{server.context.trace_id:032x}" == TRACE_ID
    assert server.parent is not None and f"{server.parent.span_id:016x}" == PARENT_ID
    # server -> gateway.proxy -> httpx client span
    assert proxy.parent is not None and proxy.parent.span_id == server.context.span_id
    assert client_span.parent is not None and client_span.parent.span_id == proxy.context.span_id
    # traceparent injected upstream carries the client span's identity
    assert seen["traceparent"] == f"00-{TRACE_ID}-{client_span.context.span_id:016x}-01"
    # health endpoints excluded
    assert not any("healthz" in str(s.attributes) for s in spans)
    # no credentials anywhere (query-string tokens on server + client spans, auth header)
    blob = rig.everything()
    assert BEARER not in blob and API_KEY not in blob
    assert "page=2" in blob  # non-sensitive query params preserved


async def test_instrument_httpx_client_for_existing_client(rig: Rig) -> None:
    from ai_gateway.observability.otel import instrument_httpx_client

    seen: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("traceparent", ""))
        return httpx.Response(204)

    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), base_url="http://up")
    instrument_httpx_client(client)
    with rig.tel.span("gateway.proxy"):
        await client.get("/x", params={"api_key": API_KEY})
    await client.aclose()
    spans = rig.spans.get_finished_spans()
    client_span = next(s for s in spans if s.kind is SpanKind.CLIENT)
    assert seen[0].split("-")[2] == f"{client_span.context.span_id:016x}"
    assert API_KEY not in rig.everything()
