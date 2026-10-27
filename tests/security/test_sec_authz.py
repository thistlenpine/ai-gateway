"""Authorization bypass attempts (path, method, scope, MCP, LLM, body parsing). Passing = secure."""

from __future__ import annotations

import json
from typing import Any

import jwt
import pytest
from sec_helpers import BILLING, LLM, MCP, call, claims, mcp_ok, post_mcp, raw_request, rpc

from ai_gateway.models import UpstreamKind

# ------------------------------------------------------------------ path normalisation


def _billing_ok(gw: Any) -> Any:
    return gw.upstream.route(host="billing.internal").respond(json={"ok": True})


HOSTILE_PATHS = [
    b"/billing/admin",
    b"/billing/invoices/../admin",
    b"/billing/invoices/%2e%2e/admin",
    b"/billing/invoices/%2E%2E/admin",
    b"/billing/invoices/.%2e/admin",
    b"/billing/invoices/%2e./admin",
    b"/billing/invoices/%2e%2e%2fadmin",
    b"/billing/invoices%2f..%2fadmin",
    b"/billing/invoices%2Fx",
    b"/billing/invoices%5c..%5cadmin",
    b"/billing/invoices\\..\\admin",
    b"/billing/invoices/%252e%252e/admin",
    b"/billing/invoices/%252fadmin",
    b"/billing/invoices%00/../admin",
    b"/billing/invoices/%00",
    b"/billing/invoices;/../admin",
    b"/billing/invoices/..;/admin",
    b"/billing/invoices%3f/../admin",
    b"/billing/invoices%23/../admin",
    b"/billing/invoices/%c0%ae%c0%ae/admin",  # overlong UTF-8 '..'
    b"/billing/invoices/%e2%80%a5/admin",  # TWO DOT LEADER
    b"/billing/invoices/%ef%bc%8e%ef%bc%8e/admin",  # fullwidth '..'
    b"/billing/invoices/%ef%bc%8fadmin",  # fullwidth solidus
    b"/billing/invoices/%u002e%u002e/admin",
    b"/billing/invoices/ ../admin",
    b"/billing/invoices/.. /admin",
    b"/billing/invoices/%20../admin",
    b"/billing/invoices/..%20/admin",
    b"/billing/invoices/%09../admin",
    b"/billing/invoices/\r\n/../admin",
    b"/billing/INVOICES/../ADMIN",
    b"/BILLING/invoices",
    b"/Billing/invoices",
    b"/billingx/invoices",
    b"/billin/g/invoices",
    b"billing/invoices",
    b"//billing/../admin",
    b"/billing/invoices/%",
    b"/billing/invoices/%zz",
    b"/billing/invoices/%f",
    b"http://evil.test/billing/invoices",
    b"/../billing/invoices",
    b"/billing/../../billing/invoices/../x",
]


@pytest.mark.parametrize(
    "raw", HOSTILE_PATHS, ids=lambda b: b.decode("latin-1").encode("unicode_escape").decode()
)
async def test_hostile_paths_never_reach_upstream_except_as_exact_allowed_path(
    make_gateway: Any, raw: bytes
) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await raw_request(
            gw.app, "GET", raw, headers=[("authorization", gw.auth("billing:read")["Authorization"])]
        )
        assert r.status_code in (400, 401, 403, 404), (raw, r.status_code)
        for call_ in gw.upstream.calls:
            # if anything was forwarded it must be exactly an allowed, fully normalised path
            assert call_.request.url.path in ("/invoices", "/invoices/42"), call_.request.url


@pytest.mark.parametrize(
    "raw", [b"/billing/invoices/\xff", b"/billing/invoices/caf\xc3\xa9", b"/billing/invoices/\x80"]
)
async def test_raw_non_ascii_request_target_rejected(make_gateway: Any, raw: bytes) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await raw_request(
            gw.app, "GET", raw, headers=[("authorization", gw.auth("billing:read")["Authorization"])]
        )
        assert r.status_code == 400 and not gw.upstream.calls


async def test_raw_non_ascii_query_is_not_a_500(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await raw_request(
            gw.app,
            "GET",
            b"/billing/invoices",
            query=b"a=\xe9\xff",
            headers=[("authorization", gw.auth("billing:read")["Authorization"])],
        )
        assert r.status_code < 500


@pytest.mark.parametrize(
    ("raw", "expected_upstream_path"),
    [
        (b"/billing/invoices/42", "/invoices/42"),
        (b"/billing/invoices/42/", "/invoices/42"),
        (b"/billing//invoices///42", "/invoices/42"),
        (b"/billing/./invoices/./42", "/invoices/42"),
        (b"/billing/x/../invoices/42", "/invoices/42"),
        (b"/billing/inv%6Fices/42", "/invoices/42"),
        (b"/billing/invoices/%34%32", "/invoices/42"),
    ],
)
async def test_forwarded_path_is_exactly_the_authorised_normalised_path(
    make_gateway: Any, raw: bytes, expected_upstream_path: str
) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await raw_request(
            gw.app, "GET", raw, headers=[("authorization", gw.auth("billing:read")["Authorization"])]
        )
        assert r.status_code == 200, raw
        assert gw.upstream.calls.last.request.url.path == expected_upstream_path
        assert gw.upstream.calls.last.request.url.host == "billing.internal"


@pytest.mark.parametrize("seg", ["@evil.test", "evil.test:80", "%40evil.test", "..%2f", "x%0d%0aHost:evil"])
async def test_userinfo_authority_tricks_in_path_cannot_change_destination(
    make_gateway: Any, seg: str
) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        h = [("authorization", gw.auth("billing:read")["Authorization"])]
        await raw_request(gw.app, "GET", f"/billing/invoices/{seg}", headers=h)
        await raw_request(gw.app, "GET", f"/billing/{seg}", headers=h)
        for c in gw.upstream.calls:
            assert c.request.url.host == "billing.internal" and c.request.url.port is None


async def test_authz_decides_on_decoded_path_not_raw_text(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        h = [("authorization", gw.auth("billing:read")["Authorization"])]
        r = await raw_request(gw.app, "GET", b"/billing/%69nvoices/%2e%2e/%61dmin", headers=h)
        assert r.status_code in (400, 403)
        assert not gw.upstream.calls


async def test_unauthenticated_never_sees_route_existence(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        a = await raw_request(gw.app, "GET", b"/billing/invoices")
        b = await raw_request(gw.app, "GET", b"/definitely/not/a/route")
        assert a.status_code == b.status_code == 401
        assert a.json()["error"]["code"] == b.json()["error"]["code"]
        for p in (
            b"/docs",
            b"/openapi.json",
            b"/redoc",
            b"/metrics",
            b"/debug",
            b"/healthz/x",
            b"/%2e%2e/healthz",
        ):
            r = await raw_request(gw.app, "GET", p)
            assert r.status_code in (401, 400, 403), p


# ------------------------------------------------------------------ method tricks


async def test_method_override_headers_cannot_upgrade_a_read_to_a_write(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        h = gw.auth("billing:read")
        for hdr in ("X-HTTP-Method-Override", "X-HTTP-Method", "X-Method-Override"):
            # POST with an override claiming GET: authz must use the real method.
            r = await gw.client.post("/billing/invoices", headers={**h, hdr: "GET"}, content=b"{}")
            assert r.status_code == 403
            # GET with an override claiming DELETE: header must not reach the upstream.
            r = await gw.client.get("/billing/invoices/42", headers={**h, hdr: "DELETE"})
            assert r.status_code == 200
            assert hdr.lower() not in gw.upstream.calls.last.request.headers
        assert all(c.request.method == "GET" for c in gw.upstream.calls)


@pytest.mark.parametrize("verb", ["get", "Get", "POSTX", "PROPFIND", "CONNECT", "TRACE", "LINK"])
async def test_odd_methods_rejected(make_gateway: Any, verb: str) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await raw_request(
            gw.app,
            verb,
            b"/billing/invoices",
            headers=[("authorization", gw.auth("billing:read")["Authorization"])],
        )
        assert r.status_code in (400, 403, 405)
        assert not gw.upstream.calls


async def test_head_not_granted_to_post_only_scope(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await gw.client.head("/billing/invoices", headers=gw.auth("nothing:here"))
        assert r.status_code == 403 and not gw.upstream.calls


# ------------------------------------------------------------------ scope matching


@pytest.mark.parametrize(
    "scopes",
    [
        "billing:*",
        "*",
        "billing",
        "billing:",
        "BILLING:READ",
        "billing:read2",
        "xbilling:read",
        "billing:read:extra",
        "billing.read",
        "billing:rea",
        "billing:read\u200b",
        "billing:r\u0435ad",
        "billing%3Aread",
        "mcp:tools:read llm:claude",
        "",
    ],
)
async def test_scope_must_match_exactly(make_gateway: Any, scopes: str) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await gw.client.get("/billing/invoices", headers=gw.auth(scopes))
        assert r.status_code == 403 and not gw.upstream.calls


async def test_scope_in_wrong_claim_or_other_claim_names_do_not_count(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        for extra in (
            {"scopes": "billing:read"},
            {"roles": ["billing:read"]},
            {"permissions": ["billing:read"]},
            {"groups": ["billing:read"]},
            {"authorities": "billing:read"},
        ):
            tok = jwt.encode(
                claims(scope=None, **extra), gw.idp.private, algorithm="RS256", headers={"kid": gw.idp.kid}
            )
            r = await gw.client.get("/billing/invoices", headers={"Authorization": f"Bearer {tok}"})
            assert r.status_code == 403, extra


async def test_scope_boundaries_with_space_separated_claim(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await gw.client.get("/billing/invoices", headers=gw.auth("other billing:read"))
        assert r.status_code == 200
        r = await gw.client.get("/billing/invoices", headers=gw.auth("billing :read"))
        assert r.status_code == 403


async def test_scope_scopes_from_scp_list_work_but_do_not_leak_across_clients(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        r = await gw.client.get("/billing/invoices", headers=gw.auth("", scp=["billing:read"]))
        assert r.status_code == 200
        r = await gw.client.get("/billing/invoices", headers=gw.auth("", client_id="client-tokens"))
        assert r.status_code == 403


async def test_method_scope_matrix_for_http_upstream(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        _billing_ok(gw)
        h = gw.auth("billing:read")
        for verb in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
            r = await gw.client.request(verb, "/billing/invoices/42", headers=h, content=b"{}")
            assert r.status_code == 403, verb
        assert not gw.upstream.calls


# ------------------------------------------------------------------ MCP


async def test_batch_hiding_denied_call_anywhere_is_denied(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        good, bad = call("get_weather"), call("delete_everything")
        for batch in (
            [good, bad],
            [bad, good],
            [good, good, bad],
            [rpc("ping"), bad],
            [rpc("notifications/initialized", id_=None), call("rm")],
        ):
            r = await post_mcp(gw, batch)
            assert r.status_code == 403, batch
        assert route.call_count == 0


async def test_batch_shapes_that_confuse_classifiers_are_rejected(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        shapes: list[Any] = [
            [],
            [[call("get_a")]],
            [call("get_a"), "x"],
            [call("get_a"), None],
            [call("get_a"), 1],
            [{"jsonrpc": "2.0", "id": 1}],
            [{"jsonrpc": "2.0", "id": 1, "result": {}}],
            [{"jsonrpc": "2.0", "id": 1, "method": None}],
            [{"jsonrpc": "2.0", "id": 1, "method": ["tools/call"]}],
            [rpc("ping")] * 101,
        ]
        for shape in shapes:
            r = await post_mcp(gw, shape)
            assert r.status_code in (400, 403), shape
        assert route.call_count == 0


@pytest.mark.parametrize(
    "msg",
    [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call"},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": None},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": None}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": ["get_a"]}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": {"x": "get_a"}}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": ""}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": 5}},
        {"jsonrpc": "2.0", "id": 1, "method": "Tools/Call", "params": {"name": "get_a"}},
        {"jsonrpc": "2.0", "id": 1, "method": " tools/call", "params": {"name": "get_a"}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call\n", "params": {"name": "get_a"}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call/", "params": {"name": "get_a"}},
        {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": "file:///etc/passwd"}},
        {"jsonrpc": "2.0", "id": 1, "method": "prompts/list"},
        {"jsonrpc": "2.0", "id": 1, "method": "sampling/createMessage"},
        {"jsonrpc": "2.0", "id": 1, "method": "logging/setLevel", "params": {"level": "debug"}},
        {"jsonrpc": "2.0", "id": 1, "method": "completion/complete"},
        {"jsonrpc": "2.0", "id": 1, "method": "get_a"},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/*"},
        {"jsonrpc": "2.0", "id": 1, "method": "*"},
    ],
)
async def test_mcp_methods_and_tool_names_default_deny(make_gateway: Any, msg: dict[str, Any]) -> None:
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        r = await post_mcp(gw, msg)
        assert r.status_code in (400, 403)
        assert route.call_count == 0


@pytest.mark.parametrize(
    "tool",
    ["get_", "get_a/../../delete_all", "search_\n", "get_\u202e", "get_%00", "get_" + "x" * 1500],
    ids=["bare", "dotdot", "newline", "rlo", "pct00", "long"],
)
async def test_allowed_glob_tool_names_are_forwarded_verbatim_not_reinterpreted(
    make_gateway: Any, tool: str
) -> None:
    """Anything matching the grant glob is allowed; what the upstream sees must be byte-identical
    to what was authorised (no normalisation drift)."""
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        r = await post_mcp(gw, call(tool))
        assert r.status_code == 200
        assert json.loads(route.calls.last.request.content)["params"]["name"] == tool


async def test_duplicate_json_keys_rejected_everywhere(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        bodies = [
            b'{"jsonrpc":"2.0","id":1,"method":"tools/list","method":"tools/call","params":{"name":"rm"}}',
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_a","name":"rm"}}',
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_a"},"params":{"name":"rm"}}',
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rm","name":"get_a"}}',
            b'{"jsonrpc":"2.0","id":1,"m\\u0065thod":"ping","method":"tools/call","params":{"name":"rm"}}',
            b'{"jsonrpc":"2.0","id":1,"method":"ping","\\u006dethod":"tools/call","params":{"name":"rm"}}',
            b'[{"jsonrpc":"2.0","id":1,"method":"ping"},{"jsonrpc":"2.0","id":1,"method":"ping","method":"tools/call"}]',
        ]
        for body in bodies:
            r = await post_mcp(gw, body)
            assert r.status_code in (400, 403), body
        assert route.call_count == 0


@pytest.mark.parametrize(
    "body",
    [
        b'{"jsonrpc":"2.0","id":1,"method":"ping","Method":"tools/call","params":{"name":"delete_all"}}',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_a","Name":"delete_all"}}',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_a"},"PARAMS":{"name":"delete_all"}}',
        b'{"jsonrpc":"2.0","id":1,"METHOD":"tools/call","method":"ping","params":{"name":"delete_all"}}',
    ],
)
async def test_case_variant_duplicate_keys_not_forwarded(make_gateway: Any, body: bytes) -> None:
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        r = await post_mcp(gw, body)
        assert route.call_count == 0, f"ambiguous body reached upstream (status {r.status_code})"


@pytest.mark.parametrize("enc", ["utf-16", "utf-16-le", "utf-32", "utf-8-sig"])
async def test_non_utf8_json_bodies_rejected(make_gateway: Any, enc: str) -> None:
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        r = await post_mcp(gw, json.dumps(call("get_a")).encode(enc))
        assert r.status_code == 400 and route.call_count == 0


async def test_body_edge_cases_are_clean_4xx_never_500(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        bodies: list[bytes] = [
            b"",
            b"null",
            b"true",
            b"1",
            b'"x"',
            b"[",
            b"{",
            b"{}",
            b"[{}]",
            b"\xff\xfe\x00",
            b"[" * 200_000,
            b'{"a":' * 100_000,
            b"[" * 1_900_000,
            b'{"jsonrpc":"2.0","id":' + b"9" * 5000 + b',"method":"ping"}',
            b'{"jsonrpc":"2.0","id":1,"method":"ping","x":NaN}',
            b'{"jsonrpc":"2.0","id":1,"method":"ping","x":Infinity}',
            b'{"jsonrpc":"2.0","id":1,"method":"ping","x":-Infinity}',
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_\\ud800"}}',
            b'{"jsonrpc":"2.0","id":1,"method":"ping"} trailing',
            b'{"jsonrpc":"2.0","id":1,"method":"ping"}{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"rm"}}',
        ]
        for body in bodies:
            r = await post_mcp(gw, body)
            assert r.status_code < 500, (body[:40], r.status_code)
        sent = [c.request.content for c in route.calls]
        assert all(b"trailing" not in s and b'"rm"' not in s for s in sent)


async def test_chunked_body_over_limit_is_413_even_without_content_length(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = mcp_ok(gw)
        limit = gw.services.config.gateway.server.max_body_bytes
        h = [("authorization", gw.auth("mcp:tools:read")["Authorization"])]
        r = await raw_request(gw.app, "POST", b"/mcp/tools", headers=h, body=[b"x" * (limit // 2 + 1)] * 2)
        assert r.status_code == 413
        r = await raw_request(
            gw.app, "POST", b"/mcp/tools", headers=[*h, ("content-length", str(limit + 1))], body=b"x"
        )
        assert r.status_code == 413
        r = await raw_request(
            gw.app, "POST", b"/mcp/tools", headers=[*h, ("content-length", "-1")], body=b"{}"
        )
        assert r.status_code == 400
        r = await raw_request(
            gw.app, "POST", b"/mcp/tools", headers=[*h, ("content-length", "1e9")], body=b"{}"
        )
        assert r.status_code == 400
        assert route.call_count == 0


def names(r: Any) -> list[str]:
    return [t["name"] for t in r["result"]["tools"]]


async def test_tools_list_filtering_json_sse_and_batch(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        tools = [{"name": n} for n in ("get_a", "search_b", "delete_all", "admin_x", "get_")]
        route = gw.upstream.post(MCP)
        for ctype in (
            "application/json",
            "application/json; charset=utf-8",
            "APPLICATION/JSON",
            "application/vnd.api+json",
        ):
            route.respond(
                json={"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}}, headers={"content-type": ctype}
            )
            r = await post_mcp(gw, rpc("tools/list"))
            assert names(r.json()) == ["get_a", "search_b", "get_"], ctype
        route.respond(
            json=[
                {"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}},
                {"jsonrpc": "2.0", "id": 2, "result": {"tools": tools}},
            ]
        )
        r = await post_mcp(gw, [rpc("tools/list"), rpc("tools/list", id_=2)])
        assert [names(x) for x in r.json()] == [["get_a", "search_b", "get_"]] * 2
        sse = (
            "event: message\r\ndata: "
            + json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}})
            + "\r\n\r\n"
        )
        route.respond(content=sse.encode(), headers={"content-type": "text/event-stream"})
        r = await post_mcp(gw, rpc("tools/list"))
        assert "delete_all" not in r.text and "admin_x" not in r.text and "get_a" in r.text


async def test_tools_list_filter_handles_malformed_entries_failing_closed(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = gw.upstream.post(MCP)
        route.respond(
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {"name": "get_a"},
                        {"name": 5},
                        {"name": None},
                        {},
                        "get_x",
                        None,
                        {"name": ["get_a"]},
                        {"name": "delete_all"},
                    ]
                },
            }
        )
        r = await post_mcp(gw, rpc("tools/list"))
        assert [t["name"] for t in r.json()["result"]["tools"]] == ["get_a"]


async def test_tools_list_sse_event_split_across_chunks_and_multiline_data(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        doc = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "get_a"}, {"name": "delete_all"}]}}
        )
        mid = len(doc) // 2
        sse = f"event: message\ndata: {doc[:mid]}\ndata: {doc[mid:]}\n\n"
        gw.upstream.post(MCP).respond(content=sse.encode(), headers={"content-type": "text/event-stream"})
        r = await post_mcp(gw, rpc("tools/list"))
        assert "delete_all" not in r.text


async def test_tools_list_filter_is_per_principal(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.post(MCP).respond(
            json={"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "get_a"}, {"name": "rm"}]}}
        )
        r = await post_mcp(gw, rpc("tools/list"), scopes="mcp:tools:read")
        assert [t["name"] for t in r.json()["result"]["tools"]] == ["get_a"]
        r = await post_mcp(gw, rpc("tools/list"), scopes="unrelated:scope")
        assert r.status_code == 403


async def test_mcp_get_and_delete_need_an_mcp_grant(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.route(host="mcp.internal").respond(json={})
        for verb in ("GET", "DELETE"):
            r = await gw.client.request(verb, "/mcp/tools", headers=gw.auth("billing:read"))
            assert r.status_code == 403, verb
        for verb in ("PUT", "PATCH", "HEAD", "OPTIONS"):
            r = await gw.client.request(verb, "/mcp/tools", headers=gw.auth("mcp:tools:read"))
            assert r.status_code in (400, 403, 405), verb
        assert not gw.upstream.calls


# ------------------------------------------------------------------ LLM


def _anthropic(gw: Any) -> Any:
    return gw.upstream.post(LLM).respond(
        json={"model": "claude-x", "usage": {"input_tokens": 1, "output_tokens": 1}}
    )


async def _llm(gw: Any, body: Any, scopes: str = "llm:claude", **kw: Any) -> Any:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return await gw.client.post("/llm/anthropic/v1/messages", headers=gw.auth(scopes, **kw), content=content)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4",
        "claude",
        "CLAUDE-3",
        " claude-3",
        "\u0441laude-3",
        "gpt-4\nclaude-3",
        "",
        None,
        5,
        ["claude-3"],
        {"id": "claude-3"},
        True,
        "*",
    ],
)
async def test_llm_model_not_granted_or_malformed(make_gateway: Any, model: Any) -> None:
    async with make_gateway() as gw:
        route = _anthropic(gw)
        r = await _llm(gw, {"model": model, "max_tokens": 5, "messages": []})
        assert r.status_code in (400, 403) and route.call_count == 0


async def test_llm_model_missing_wrong_shape_or_duplicated(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _anthropic(gw)
        for body in (
            b"{}",
            b"[]",
            b'"claude-3"',
            b"null",
            b'{"model":"claude-3","model":"gpt-4"}',
            b'{"model":"gpt-4","model":"claude-3"}',
            b'{"mod\\u0065l":"gpt-4","model":"claude-3"}',
        ):
            r = await _llm(gw, body)
            assert r.status_code in (400, 403), body
        assert route.call_count == 0


async def test_llm_case_variant_model_key_not_forwarded(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _anthropic(gw)
        await _llm(gw, b'{"model":"claude-3","Model":"gpt-4","max_tokens":5}')
        assert route.call_count == 0


async def test_llm_scope_to_model_matrix(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _anthropic(gw)
        assert (
            await _llm(gw, {"model": "claude-3", "max_tokens": 5}, scopes="billing:read")
        ).status_code == 403
        assert (
            await _llm(gw, {"model": "claude-3", "max_tokens": 5}, scopes="mcp:tools:read")
        ).status_code == 403
        assert route.call_count == 0
        assert (await _llm(gw, {"model": "claude-3", "max_tokens": 5})).status_code == 200


async def test_llm_get_models_listing_not_openly_reachable(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.get("http://llm.internal/v1/models").respond(json={"data": []})
        r = await gw.client.get("/llm/anthropic/v1/models", headers=gw.auth("billing:read"))
        assert r.status_code == 403 and not gw.upstream.calls


@pytest.mark.parametrize(
    "tokens", [None, 10**9, 10**30, -1, 0, "100000", 1e9, 1e400, True, [1], {"a": 1}, 8193, 2**63]
)
async def test_llm_max_tokens_always_clamped_to_grant_cap(make_gateway: Any, tokens: Any) -> None:
    async with make_gateway() as gw:
        route = _anthropic(gw)
        body: dict[str, Any] = {"model": "claude-3", "messages": []}
        if tokens is not None:
            body["max_tokens"] = tokens
        raw = json.dumps(body).replace("Infinity", "1e999").encode()
        r = await _llm(gw, raw)
        if r.status_code == 200:
            sent = json.loads(route.calls.last.request.content)
            assert isinstance(sent["max_tokens"], int) and 0 < sent["max_tokens"] <= 8192


async def test_llm_user_supplied_stream_options_and_extra_fields_do_not_bypass_cap(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _anthropic(gw)
        r = await _llm(
            gw,
            {
                "model": "claude-3",
                "max_tokens": 10**7,
                "max_completion_tokens": 10**7,
                "max_output_tokens": 10**7,
                "stream": False,
            },
        )
        assert r.status_code == 200
        sent = json.loads(route.calls.last.request.content)
        for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            assert sent[k] <= 8192


# ------------------------------------------------------------------ misc hardening of the 4xx/5xx surface


async def test_internal_errors_are_generic_and_traceless(make_gateway: Any) -> None:
    async with make_gateway() as gw:

        async def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("SECRET-INTERNAL-DETAIL /etc/shadow postgres://u:pw@db/x")

        gw.services.validator.validate = boom
        r = await raw_request(
            gw.app, "GET", b"/billing/invoices", headers=[("authorization", "Bearer a.b.c")]
        )
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "internal_error"
        assert "SECRET-INTERNAL" not in r.text and "Traceback" not in r.text and "shadow" not in r.text
        assert "SECRET-INTERNAL" not in gw.obs.everything() or "pw@" not in gw.obs.everything()


async def test_unauthenticated_and_unauthorised_bodies_are_uniform(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        seen = set()
        for path in ("/billing/invoices", "/nope", "/mcp/tools", "/llm/anthropic/v1/messages"):
            r = await raw_request(gw.app, "GET", path.encode(), headers=[("authorization", "Bearer x.y.z")])
            seen.add((r.status_code, r.json()["error"]["code"], r.json()["error"]["message"]))
        assert len(seen) == 1


async def test_constants_used(make_gateway: Any) -> None:
    assert BILLING.startswith("http")


# ------------------------------------------------------------------ oracles and ordering


async def test_route_existence_is_not_an_oracle_for_callers_without_grants(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        h = gw.auth("billing:read")  # no MCP / LLM scope at all
        seen = set()
        for path in ("/mcp/tools", "/llm/anthropic/v1/messages", "/no/such/route", "/billing/admin"):
            r = await gw.client.post(path, headers=h, content=b"{not json")
            seen.add((r.status_code, r.json()["error"]["code"]))
        assert len(seen) == 1, seen


async def test_ungranted_caller_body_is_not_buffered(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        calls = 0
        orig = gw.services.proxy.adapter(UpstreamKind.MCP).classify

        def spy(*a: Any, **k: Any) -> Any:
            nonlocal calls
            calls += 1
            return orig(*a, **k)

        gw.services.proxy.adapter(UpstreamKind.MCP).classify = spy
        await gw.client.post("/mcp/tools", headers=gw.auth("billing:read"), content=json.dumps(call("get_a")))
        assert calls == 0


async def test_idp_down_with_cold_cache_fails_closed_without_touching_upstreams(make_gateway: Any) -> None:
    from ai_gateway.auth import TokenValidator

    async with make_gateway() as gw:
        gw.upstream.route(host="billing.internal").respond(json={})
        tok = gw.token("billing:read")
        import respx

        down = respx.Router(assert_all_called=False)
        down.route().respond(status_code=500)
        import httpx

        gw.services.validator = TokenValidator(
            gw.services.config.gateway.idps,
            httpx.AsyncClient(transport=httpx.MockTransport(down.async_handler)),
            gw.services.telemetry,
        )
        r = await raw_request(
            gw.app, "GET", b"/billing/invoices", headers=[("authorization", f"Bearer {tok}")]
        )
        assert r.status_code == 503 and r.json()["error"]["code"] == "idp_unavailable"
        assert not gw.upstream.calls
        assert (await raw_request(gw.app, "GET", b"/readyz")).status_code == 503


@pytest.mark.parametrize(
    "claim,value",
    [
        ("iss", ["https://idp.test"]),
        ("iss", {"a": 1}),
        ("iss", 5),
        ("aud", {"a": "ai-gateway"}),
        ("aud", 5),
        ("aud", [["ai-gateway"]]),
        ("sub", ["x"]),
        ("sub", 5),
    ],
)
async def test_wrong_typed_registered_claims_are_401_not_500(
    make_gateway: Any, claim: str, value: Any
) -> None:
    async with make_gateway() as gw:
        gw.upstream.route(host="billing.internal").respond(json={})
        tok = jwt.api_jws.PyJWS().encode(
            json.dumps(claims(**{claim: value})).encode(),
            gw.idp.private,
            algorithm="RS256",
            headers={"kid": gw.idp.kid},
        )
        r = await raw_request(
            gw.app, "GET", b"/billing/invoices", headers=[("authorization", f"Bearer {tok}")]
        )
        assert r.status_code == 401, (claim, value, r.status_code)


async def test_disabled_client_is_denied_even_with_a_valid_token(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.route(host="billing.internal").respond(json={})
        gw.services.config.clients.clients["client-a"].enabled = False
        r = await gw.client.get("/billing/invoices", headers=gw.auth("billing:read"))
        assert r.status_code == 403 and not gw.upstream.calls


async def test_multiple_authorization_headers_cannot_pick_the_weaker_validator(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.route(host="billing.internal").respond(json={})
        good = gw.auth("billing:read")["Authorization"]
        for order in ((good, "Bearer garbage"), ("Bearer garbage", good)):
            r = await raw_request(
                gw.app, "GET", b"/billing/invoices", headers=[("authorization", v) for v in order]
            )
            if r.status_code == 200:  # first header wins; the other one must never be consulted or forwarded
                sent = gw.upstream.calls.last.request.headers
                assert "authorization" not in sent
