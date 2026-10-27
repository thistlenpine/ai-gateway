"""Proxy hygiene: credential leakage, header spoofing/smuggling, SSRF, response handling."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import jwt
import pytest
from sec_helpers import LLM, MCP, claims, mcp_ok, post_mcp, raw_request, rpc

from gw_rig import LLM_SECRET, MCP_SECRET


def _ok(gw: Any, **kw: Any) -> Any:
    return gw.upstream.route(host="billing.internal").respond(json={"ok": True}, **kw)


def _bearer(gw: Any, scopes: str = "billing:read", **kw: Any) -> tuple[str, str]:
    return ("authorization", gw.auth(scopes, **kw)["Authorization"])


# ------------------------------------------------------------------ caller credentials never go upstream


async def test_caller_credentials_and_proxy_auth_headers_never_forwarded(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        tok = gw.token("billing:read")
        hdrs = [
            ("authorization", f"Bearer {tok}"),
            ("Authorization", "Bearer second-token"),
            ("proxy-authorization", "Basic abc"),
            ("cookie", "session=s3cr3t"),
            ("x-api-key", "caller-key"),
            ("api-key", "caller-key"),
            ("x-goog-api-key", "caller-key"),
            ("x-auth-token", "caller-token"),
        ]
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=hdrs)
        assert r.status_code == 200
        sent = gw.upstream.calls.last.request
        wire = repr(sent.headers.raw) + repr(sent.url)
        for needle in (tok, "second-token", "Basic abc", "s3cr3t", "caller-key", "caller-token"):
            assert needle not in wire, needle
        for name in (
            "authorization",
            "proxy-authorization",
            "cookie",
            "x-api-key",
            "api-key",
            "x-goog-api-key",
        ):
            assert name not in sent.headers


async def test_token_in_query_string_is_not_an_alternative_auth_path(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        tok = gw.token("billing:read")
        for q in (f"access_token={tok}", f"token={tok}", f"api_key={tok}"):
            r = await raw_request(gw.app, "GET", b"/billing/invoices", query=q)
            assert r.status_code == 401, q


@pytest.mark.parametrize(
    "hdr",
    [
        "X-Gateway-Client-Id",
        "x-gateway-subject",
        "X-Gateway-Anything",
        "X-Forwarded-For",
        "X-Forwarded-Host",
        "X-Forwarded-Proto",
        "X-Forwarded-Prefix",
        "X-Forwarded-User",
        "Forwarded",
        "Via",
        "X-Real-IP",
        "X-Client-IP",
        "True-Client-IP",
        "CF-Connecting-IP",
        "X-Original-URL",
        "X-Original-URI",
        "X-Rewrite-URL",
        "X-Envoy-Internal",
        "X-Amzn-Trace-Id",
        "Proxy-Foo",
        "Traceparent",
        "Baggage",
    ],
)
async def test_identity_and_proxy_headers_cannot_be_spoofed_upstream(make_gateway: Any, hdr: str) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw), (hdr, "admin")])
        assert r.status_code == 200
        sent = gw.upstream.calls.last.request.headers
        if hdr.lower() == "x-gateway-client-id":
            assert sent["x-gateway-client-id"] == "client-a"
        elif hdr.lower() == "x-gateway-subject":
            assert sent["x-gateway-subject"] == "client-a"
        elif hdr.lower() in ("traceparent",):
            assert sent.get("traceparent") != "admin"
        else:
            assert sent.get(hdr.lower()) != "admin", hdr
        assert [v for k, v in sent.multi_items() if v == "admin"] == []


async def test_identity_headers_always_come_from_the_token(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        h = [
            _bearer(gw, client_id="client-a"),
            ("x-gateway-client-id", "victim"),
            ("X-GATEWAY-SUBJECT", "root"),
        ]
        await raw_request(gw.app, "GET", b"/billing/invoices", headers=h)
        sent = gw.upstream.calls.last.request.headers
        assert sent.get_list("x-gateway-client-id") == ["client-a"]
        assert sent.get_list("x-gateway-subject") == ["client-a"]


async def test_hostile_sub_claim_cannot_inject_headers(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        for sub in ("a\r\nX-Evil: 1", "a\nb", "é\u0000x", "x" * 10_000, "‮"):
            tok = jwt.encode(
                claims(sub=sub, client_id="client-a", scope="billing:read"),
                gw.idp.private,
                algorithm="RS256",
                headers={"kid": gw.idp.kid},
            )
            r = await raw_request(
                gw.app, "GET", b"/billing/invoices", headers=[("authorization", f"Bearer {tok}")]
            )
            if r.status_code == 200:
                sent = gw.upstream.calls.last.request.headers
                assert "x-evil" not in sent
                for k, v in sent.multi_items():
                    assert "\r" not in v and "\n" not in v and "\x00" not in v, (k, v)


# ------------------------------------------------------------------ header smuggling


async def test_hop_by_hop_and_connection_listed_headers_are_stripped(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        h = [
            _bearer(gw),
            ("connection", "close, X-Smuggle, x-gateway-client-id"),
            ("x-smuggle", "1"),
            ("keep-alive", "timeout=5"),
            ("te", "trailers"),
            ("trailer", "x"),
            ("upgrade", "websocket"),
            ("proxy-connection", "keep-alive"),
            ("transfer-encoding", "chunked"),
            ("expect", "100-continue"),
        ]
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=h)
        assert r.status_code == 200
        sent = gw.upstream.calls.last.request.headers
        for n in ("x-smuggle", "keep-alive", "te", "trailer", "upgrade", "proxy-connection", "expect"):
            assert n not in sent, n
        assert not ("transfer-encoding" in sent and "content-length" in sent)


async def test_crlf_and_control_chars_in_header_names_values_are_dropped(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        h = [
            _bearer(gw),
            ("x-a", "v\r\nX-Injected: 1"),
            ("x-b", "v\nX-Injected2: 1"),
            ("x-c", "v\x00w"),
            ("x b", "space-in-name"),
            ("x:colon", "v"),
            ("x-d\r\nx-e", "v"),
            ("x-ok", "fine\tvalue"),
        ]
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=h)
        assert r.status_code == 200
        sent = gw.upstream.calls.last.request.headers
        assert "x-injected" not in sent and "x-injected2" not in sent
        assert "x-a" not in sent and "x-b" not in sent and "x-c" not in sent
        assert sent.get("x-ok") == "fine\tvalue"


async def test_request_id_cannot_carry_injection(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        for rid in ("abc\r\nX-Evil: 1", "short", "a" * 129, "has space here", "ok-request-id-1\n", "é" * 10):
            r = await raw_request(
                gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw), ("x-request-id", rid)]
            )
            assert r.headers["x-request-id"] != rid
            assert "\r" not in r.headers["x-request-id"] and "\n" not in r.headers["x-request-id"]
            assert gw.upstream.calls.last.request.headers["x-request-id"] == r.headers["x-request-id"]


async def test_duplicate_content_length_and_te_cl_combination_do_not_reach_upstream_as_smuggling(
    make_gateway: Any,
) -> None:
    async with make_gateway() as gw:
        gw.upstream.post("http://billing.internal/invoices").respond(json={})
        h = [_bearer(gw, "billing:write")]
        r = await raw_request(
            gw.app,
            "POST",
            b"/billing/invoices",
            headers=[*h, ("content-length", "2"), ("content-length", "50"), ("transfer-encoding", "chunked")],
            body=b"{}",
        )
        sent = gw.upstream.calls
        assert r.status_code in (400, 403, 200)
        for c in sent:
            heads = c.request.headers
            assert not ("transfer-encoding" in heads and "content-length" in heads)
            assert len(heads.get_list("content-length")) <= 1


# ------------------------------------------------------------------ upstream credential handling


async def test_upstream_credential_replaces_caller_supplied_value(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        mcp = mcp_ok(gw)
        body = json.dumps(rpc("ping")).encode()
        r = await raw_request(
            gw.app,
            "POST",
            b"/mcp/tools",
            body=body,
            headers=[
                ("authorization", gw.auth("mcp:tools:read")["Authorization"]),
                ("authorization", "Bearer second"),
                ("X-API-Key", "caller"),
                ("content-type", "application/json"),
            ],
        )
        assert r.status_code == 200
        sent = mcp.calls.last.request.headers
        assert sent.get_list("authorization") == [f"Bearer {MCP_SECRET}"]
        assert "x-api-key" not in sent

        llm = gw.upstream.post(LLM).respond(
            json={"model": "claude-3", "usage": {"input_tokens": 1, "output_tokens": 1}}
        )
        r = await gw.client.post(
            "/llm/anthropic/v1/messages",
            headers={**gw.auth("llm:claude"), "X-API-KEY": "caller-key", "anthropic-version": "2023-06-01"},
            content=json.dumps({"model": "claude-3", "max_tokens": 5}),
        )
        assert r.status_code == 200
        sent = llm.calls.last.request.headers
        assert sent.get_list("x-api-key") == [LLM_SECRET]
        assert "authorization" not in sent


async def test_upstream_credential_never_appears_in_any_response_or_telemetry(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.post(MCP).respond(json={"jsonrpc": "2.0", "id": 1, "result": {}})
        gw.upstream.post(LLM).respond(status_code=500, json={"error": "boom"})
        out = []
        out.append((await post_mcp(gw, rpc("ping"))).text)
        out.append((await post_mcp(gw, rpc("tools/call", {"name": "nope"}))).text)
        r = await gw.client.post(
            "/llm/anthropic/v1/messages",
            headers=gw.auth("llm:claude"),
            content=json.dumps({"model": "claude-3", "max_tokens": 5}),
        )
        out.append(r.text + repr(r.headers))
        blob = "\n".join(out) + gw.obs.everything()
        assert MCP_SECRET not in blob and LLM_SECRET not in blob


async def test_missing_credential_at_runtime_is_a_generic_502(
    make_gateway: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with make_gateway() as gw:
        mcp_ok(gw)
        monkeypatch.delenv("E2E_MCP_TOKEN")
        r = await post_mcp(gw, rpc("ping"))
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "upstream_error"
        assert "E2E_MCP_TOKEN" not in r.text and "env" not in r.text.lower()
        assert not gw.upstream.calls  # never sent without credential


# ------------------------------------------------------------------ SSRF / redirects


async def test_upstream_redirects_are_not_followed(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.get("http://billing.internal/invoices").respond(
            status_code=302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}
        )
        meta = gw.upstream.get("http://169.254.169.254/latest/meta-data/").respond(text="secret")
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw)])
        assert r.status_code == 302 and meta.call_count == 0
        assert "secret" not in r.text


async def test_internal_upstream_location_not_leaked_to_clients(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.get("http://billing.internal/invoices").respond(
            status_code=307, headers={"Location": "http://billing.internal/invoices/"}
        )
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw)])
        assert "billing.internal" not in r.headers.get("location", "")


async def test_client_cannot_choose_upstream_host_via_any_header(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _ok(gw)
        h = [
            _bearer(gw),
            ("host", "evil.test"),
            ("x-forwarded-host", "evil.test"),
            ("x-original-url", "http://evil.test/"),
            ("x-forwarded-server", "evil.test"),
            ("forwarded", "host=evil.test"),
        ]
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=h)
        assert r.status_code == 200
        req = gw.upstream.calls.last.request
        assert req.url.host == "billing.internal"
        assert req.headers["host"] == "billing.internal"
        assert "evil.test" not in repr(req.headers.raw)


# ------------------------------------------------------------------ response side


async def test_response_splitting_and_framing_headers_from_upstream_are_neutralised(
    make_gateway: Any,
) -> None:
    async with make_gateway() as gw:
        gw.upstream.get("http://billing.internal/invoices").mock(
            return_value=httpx.Response(
                200,
                content=b"{}",
                headers=[
                    ("X-Ok", "fine"),
                    ("Content-Length", "3"),
                    ("Transfer-Encoding", "chunked"),
                    ("Connection", "X-Hop, close"),
                    ("X-Hop", "secret"),
                    ("Keep-Alive", "5"),
                    ("X-Request-ID", "upstream-spoof-id"),
                    ("RateLimit-Remaining", "99999"),
                ],
            )
        )
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw)])
        h = r.headers
        assert h["x-ok"] == "fine"
        assert "x-hop" not in h and "keep-alive" not in h and "transfer-encoding" not in h
        assert h["x-request-id"] != "upstream-spoof-id"
        assert h["ratelimit-remaining"] != "99999"


async def test_upstream_set_cookie_not_relayed(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.get("http://billing.internal/invoices").respond(
            200, json={}, headers={"Set-Cookie": "sid=attacker; Domain=gateway.test; Path=/"}
        )
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw)])
        assert "set-cookie" not in r.headers


async def test_utf8_response_header_does_not_wedge_accounting(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = gw.upstream.get("http://billing.internal/invoices")
        route.mock(
            return_value=httpx.Response(
                200, content=b"{}", headers=cast(Any, [("X-Filename", "résumé-☃.pdf".encode())])
            )
        )
        closed: list[bool] = []
        orig = gw.services.proxy.forward

        async def spy(*a: Any, **k: Any) -> Any:
            resp = await orig(*a, **k)
            inner = resp._close

            async def close() -> None:
                closed.append(True)
                if inner:
                    await inner()

            resp._close = close
            return resp

        gw.services.proxy.forward = spy
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw)])
        assert closed, "upstream response was never closed"
        assert r.status_code in (200, 502)


async def test_llm_token_reservation_not_leaked_when_response_cannot_be_built(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.post(LLM).mock(
            return_value=httpx.Response(
                200,
                json={"usage": {"input_tokens": 1, "output_tokens": 1}},
                headers=cast(Any, [("X-Model-Note", "☃".encode())]),
            )
        )
        body = json.dumps({"model": "claude-3", "max_tokens": 100}).encode()
        h = {**gw.auth("llm:claude", client_id="client-tokens")}
        await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=body)
        store = gw.services.store
        keys = [k for k in store._data if ":tok:" in k]
        # response is served (bad header dropped) and billed at actual usage (2), never the 100 reserved
        assert all(store._data[k][0] <= 2 for k in keys)


async def test_upstream_errors_are_generic_for_connect_timeout_and_garbage(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        for exc in (
            httpx.ConnectError("10.1.2.3:5432 refused"),
            httpx.ReadTimeout("slow"),
            httpx.RemoteProtocolError("bad framing from billing.internal"),
            httpx.InvalidURL("x"),
        ):
            gw.upstream.get("http://billing.internal/invoices").mock(side_effect=exc)
            r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw)])
            assert r.status_code in (502, 504)
            assert "10.1.2.3" not in r.text and "billing.internal" not in r.text and "refused" not in r.text


async def test_upstream_response_too_large_is_bounded(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.services.proxy._max_response = 1000
        gw.upstream.get("http://billing.internal/invoices").respond(200, content=b"x" * 5000)
        r = await raw_request(gw.app, "GET", b"/billing/invoices", headers=[_bearer(gw)])
        assert len(r.content) <= 1000 or r.status_code >= 500


async def test_http_request_body_streaming_is_bounded(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.post("http://billing.internal/invoices").respond(json={})
        limit = gw.services.config.gateway.server.max_body_bytes
        r = await raw_request(
            gw.app,
            "POST",
            b"/billing/invoices",
            headers=[_bearer(gw, "billing:write")],
            body=[b"x" * (limit // 2)] * 3,
        )
        sent = sum(len(c.request.content) for c in gw.upstream.calls)
        assert r.status_code in (413, 502, 500) or sent <= limit
        assert sent <= limit


async def test_mcp_session_header_passthrough_documented(make_gateway: Any) -> None:
    """Mcp-Session-Id is forwarded untouched (spec requirement); the upstream sees the *current*
    caller's X-Gateway-Client-Id, which is what it must use to bind sessions to principals."""
    async with make_gateway() as gw:
        mcp = mcp_ok(gw)
        r = await post_mcp(gw, rpc("ping"), headers={"Mcp-Session-Id": "sess-123"}, **{})
        assert r.status_code == 200
        sent = mcp.calls.last.request.headers
        assert sent["mcp-session-id"] == "sess-123" and sent["x-gateway-client-id"] == "client-a"


async def test_prefix_root_request_is_not_given_a_trailing_slash(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.post("http://mcp.internal/mcp").respond(json={"jsonrpc": "2.0", "id": 1, "result": {}})
        gw.upstream.post("http://mcp.internal/mcp/").respond(
            status_code=307, headers={"location": "http://mcp.internal/mcp"}
        )
        r = await post_mcp(gw, rpc("ping"))
        assert r.status_code == 200, "MCP root was requested as .../mcp/ and got redirected"
