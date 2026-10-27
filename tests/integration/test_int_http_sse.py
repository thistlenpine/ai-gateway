"""Real gateway + real upstream over real sockets: HTTP passthrough and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from integration.servers import UpstreamState

SLEEP_STEP = 0.02


async def _until(pred: Any, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "condition not reached in time"
        await asyncio.sleep(SLEEP_STEP)


async def test_plain_http_roundtrip_strips_caller_credentials_and_adds_identity(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState
) -> None:
    gw = make_gw()
    h = {**gw.headers("api:read"), "Cookie": "session=secret", "X-Gateway-Client-Id": "spoofed"}
    r = await http.get(f"{gw.url}/api/echo/a/b?x=1&y=two", headers=h)
    assert r.status_code == 200
    seen = r.json()
    assert seen["path"] == "/api/echo/a/b" and seen["query"] == "x=1&y=two"
    up = {k.lower(): v for k, v in seen["headers"].items()}
    assert "authorization" not in up and "cookie" not in up  # caller token never forwarded
    assert up["x-gateway-client-id"] == "int-client"  # inbound spoof replaced by the verified value
    assert up["x-request-id"] == r.headers["x-request-id"]
    assert r.headers["ratelimit-limit"] == "1000"


async def test_traversal_is_normalized_before_forwarding(make_gw: Any, http: httpx.AsyncClient) -> None:
    gw = make_gw()
    r = await http.get(f"{gw.url}/api/echo/../echo/ok", headers=gw.headers())
    # httpx normalizes dot segments client side; send raw to exercise the gateway
    reader, writer = await asyncio.open_connection("127.0.0.1", gw.server.port)
    tok = gw.headers()["Authorization"]
    writer.write(
        f"GET /api/x/%2e%2e/echo/ok HTTP/1.1\r\nHost: g\r\nAuthorization: {tok}\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    raw = await reader.read()
    writer.close()
    assert raw.startswith(b"HTTP/1.1 403") or raw.startswith(b"HTTP/1.1 400")  # encoded dot segment: rejected
    assert r.status_code == 200


async def test_sse_is_relayed_incrementally_not_buffered(make_gw: Any, http: httpx.AsyncClient) -> None:
    gw = make_gw()
    arrivals: list[float] = []
    events: list[dict[str, Any]] = []
    async with http.stream("GET", f"{gw.url}/api/sse?n=6&interval=0.25", headers=gw.headers()) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "content-length" not in r.headers
        t0 = time.monotonic()
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                raw, buf = buf.split("\n\n", 1)
                data = next(line[6:] for line in raw.splitlines() if line.startswith("data: "))
                events.append(json.loads(data))
                arrivals.append(time.monotonic() - t0)
    assert [e["i"] for e in events] == list(range(6))
    # first event must show up long before the 1.25s the whole stream takes
    assert arrivals[0] < 0.5 and arrivals[-1] > 1.0
    assert all(b >= a for a, b in zip(arrivals, arrivals[1:], strict=False))


async def test_client_disconnect_mid_stream_closes_upstream_and_releases_state(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState
) -> None:
    gw = make_gw(limits={"requests_per_window": 10, "delay_start": 0.1, "max_delay_ms": 100})
    # push the client into the delay band so the request holds a delay slot while streaming
    for _ in range(2):
        await http.get(f"{gw.url}/api/echo", headers=gw.headers())
    hdr = gw.headers()
    async with http.stream("GET", f"{gw.url}/api/sse?n=1000&interval=0.05", headers=hdr) as r:
        async for _ in r.aiter_bytes():
            break  # got the first bytes: now drop the connection
    await _until(lambda: upstream_state.sse_open == 0, 5)
    assert upstream_state.sse_aborted == 1 and upstream_state.sse_completed == 0
    assert not gw.store_keys("rl:int-client:delayq")  # delay slot freed
    # and the gateway keeps serving
    assert (await http.get(f"{gw.url}/api/echo", headers=gw.headers())).status_code == 200


async def test_streamed_request_body_is_forwarded_chunked(make_gw: Any, http: httpx.AsyncClient) -> None:
    gw = make_gw()

    async def body() -> AsyncIterator[bytes]:
        for _ in range(50):
            yield b"x" * 1000
            await asyncio.sleep(0)

    r = await http.post(f"{gw.url}/api/upload", content=body(), headers=gw.headers("api:write"))
    assert r.status_code == 200 and r.json() == {"received": 50_000}


async def test_error_paths_over_real_sockets(make_gw: Any, http: httpx.AsyncClient) -> None:
    gw = make_gw()
    assert (await http.get(f"{gw.url}/healthz")).status_code == 200
    assert (await http.get(f"{gw.url}/readyz")).json()["status"] == "ready"  # JWKS fetched from the real IdP
    no_auth = await http.get(f"{gw.url}/api/echo")
    assert no_auth.status_code == 401 and "www-authenticate" in no_auth.headers
    denied = await http.post(f"{gw.url}/api/echo", headers=gw.headers("api:read"))
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "access_denied"
    assert (await http.get(f"{gw.url}/nope", headers=gw.headers())).status_code == 403
    rid = "trace-me-0123456789"
    r = await http.get(f"{gw.url}/api/echo", headers={**gw.headers(), "X-Request-ID": rid})
    assert r.headers["x-request-id"] == rid


async def test_real_time_phases_over_the_wire(make_gw: Any, http: httpx.AsyncClient) -> None:
    gw = make_gw(
        limits={"requests_per_window": 10, "window_seconds": 60, "delay_start": 0.5, "max_delay_ms": 200,
                "block_at": 1.5, "block_sustained_seconds": 0, "block_duration_seconds": 30}
    )  # fmt: skip
    codes, durations = [], []
    for _ in range(16):
        t = time.monotonic()
        r = await http.get(f"{gw.url}/api/echo", headers=gw.headers())
        durations.append(time.monotonic() - t)
        codes.append(r.status_code)
    assert codes[:10] == [200] * 10 and set(codes[10:14]) == {429}
    assert max(durations[5:10]) > 0.03  # phase 1: visibly delayed
    assert max(durations[:4]) < 0.1
    # ratio >= block_at (15/10) with sustained=0 blocks immediately
    r = await http.get(f"{gw.url}/api/echo", headers=gw.headers())
    assert r.status_code == 429 and r.json()["error"]["code"] == "client_blocked"
    assert 1 <= int(r.headers["retry-after"]) <= 30
    other = await http.get(f"{gw.url}/api/echo", headers=gw.headers(client_id="someone-else"))
    assert other.status_code == 200
