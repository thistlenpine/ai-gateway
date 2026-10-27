"""MCP, LLM accounting, phased rate limiting, isolation and observability end to end."""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx

from gw_rig import LLM_SECRET, MCP_SECRET

MCP = "http://mcp.internal/mcp"
LLM = "http://llm.internal/v1/messages"


def _rpc(method: str, params: dict[str, Any] | None = None) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()


def _anthropic_ok(gw: Any, inp: int = 10, out: int = 20) -> respx.Route:
    route: respx.Route = gw.upstream.post(LLM).respond(
        json={"id": "m1", "model": "claude-sonnet-4", "usage": {"input_tokens": inp, "output_tokens": out}}
    )
    return route


# ------------------------------------------------------------------ MCP


async def test_mcp_tool_allowed_and_denied(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = gw.upstream.post(MCP).respond(json={"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
        h = gw.auth("mcp:tools:read")
        ok = await gw.client.post(
            "/mcp/tools", headers=h, content=_rpc("tools/call", {"name": "get_weather", "arguments": {}})
        )
        assert ok.status_code == 200 and ok.json()["result"] == {"ok": True}
        assert route.calls.last.request.headers["authorization"] == f"Bearer {MCP_SECRET}"

        denied = await gw.client.post(
            "/mcp/tools", headers=h, content=_rpc("tools/call", {"name": "delete_everything"})
        )
        assert denied.status_code == 403 and denied.json()["error"]["code"] == "access_denied"
        assert route.call_count == 1  # the denied call never reached the upstream
        ev = [e for e in gw.obs.audit("authz.decision") if e["decision"] == "deny"]
        assert ev and ev[-1]["tool"] == "delete_everything"

        batch = await gw.client.post(
            "/mcp/tools",
            headers=h,
            content=json.dumps(
                [
                    json.loads(_rpc("tools/call", {"name": "get_a"})),
                    json.loads(_rpc("tools/call", {"name": "rm"})),
                ]
            ),
        )
        assert batch.status_code == 403  # every message in a batch must be allowed

        bad = await gw.client.post("/mcp/tools", headers=h, content=b"{not json")
        assert bad.status_code == 400


async def test_mcp_tools_list_is_filtered(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        tools = [{"name": "get_a"}, {"name": "search_b"}, {"name": "drop_table"}]
        gw.upstream.post(MCP).respond(json={"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}})
        r = await gw.client.post("/mcp/tools", headers=gw.auth("mcp:tools:read"), content=_rpc("tools/list"))
        assert [t["name"] for t in r.json()["result"]["tools"]] == ["get_a", "search_b"]


# ------------------------------------------------------------------ LLM


def _msg(max_tokens: int = 100, model: str = "claude-sonnet-4", **kw: Any) -> bytes:
    return json.dumps(
        {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": "hi"}], **kw}
    ).encode()


async def test_llm_accounting_and_token_reconcile(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _anthropic_ok(gw)
        h = gw.auth("llm:claude", client_id="client-tokens")  # quota: 200 tokens / window
        # each request reserves ~101 (max_tokens + prompt estimate); without reconciling to the
        # real usage (30) the second one would exceed the 200 quota.
        for _ in range(3):
            r = await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=_msg())
            assert r.status_code == 200, r.text
        assert route.calls.last.request.headers["x-api-key"] == LLM_SECRET
        points = gw.obs.metric("gateway.llm.tokens")
        by_type = {p[0]["gen_ai.token.type"]: p[1].value for p in points}
        assert by_type == {"input": 30, "output": 60}
        assert all(p[0]["gen_ai.request.model"] == "claude-sonnet-4" for p in points)

        too_big = await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=_msg(max_tokens=500))
        assert too_big.status_code == 429
        assert too_big.json()["error"]["code"] == "token_quota_exceeded"
        assert "retry-after" in too_big.headers


async def test_llm_denied_model_and_max_tokens_clamp(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        route = _anthropic_ok(gw)
        h = gw.auth("llm:claude")
        denied = await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=_msg(model="gpt-4"))
        assert denied.status_code == 403
        no_model = await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=b'{"messages":[]}')
        assert no_model.status_code == 400
        assert route.call_count == 0
        ok = await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=_msg(max_tokens=99999))
        assert ok.status_code == 200
        assert json.loads(route.calls.last.request.content)["max_tokens"] == 8192


async def test_llm_streaming_usage_and_upstream_failure_refund(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        sse = (
            'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4",'
            '"usage":{"input_tokens":7,"output_tokens":1}}}\n\n'
            'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'
        )
        gw.upstream.post(LLM).respond(content=sse, headers={"content-type": "text/event-stream"})
        h = gw.auth("llm:claude", client_id="client-tokens")
        r = await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=_msg(stream=True))
        assert r.status_code == 200 and "message_delta" in r.text
        by_type = {p[0]["gen_ai.token.type"]: p[1].value for p in gw.obs.metric("gateway.llm.tokens")}
        assert by_type == {"input": 7, "output": 9}

        # An upstream failure must refund the reservation: otherwise 3 failures exhaust 200 tokens.
        gw.upstream.routes.clear()
        gw.upstream.post(LLM).mock(side_effect=httpx.ConnectError("down"))
        for _ in range(4):
            bad = await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=_msg())
            assert bad.status_code == 502
        gw.upstream.routes.clear()
        _anthropic_ok(gw)
        again = await gw.client.post("/llm/anthropic/v1/messages", headers=h, content=_msg())
        assert again.status_code == 200


# ------------------------------------------------------------------ phased rate limiting


async def _get(gw: Any, headers: dict[str, str]) -> httpx.Response:
    r: httpx.Response = await gw.client.get("/billing/invoices/42", headers=headers)
    return r


async def test_three_phases_delay_reject_block_recover(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        up = gw.upstream.get("http://billing.internal/invoices/42").respond(json={"ok": True})
        h = gw.auth("billing:read")

        # phase 0: ratio < 0.5 -> no delay
        for _ in range(4):
            assert (await _get(gw, h)).status_code == 200
        assert gw.sleeps == []

        # phase 1: requests 5..10 are admitted after a (bounded) delay
        for _ in range(6):
            assert (await _get(gw, h)).status_code == 200
        # delay grows linearly from 0 (at delay_start) to max_delay_ms (at reject_at); the first is 0
        assert len(gw.sleeps) == 5 and all(0 < s <= 1.0 for s in gw.sleeps)
        assert gw.sleeps == sorted(gw.sleeps)
        assert [e for e in gw.obs.audit("ratelimit.decision") if e["decision"] == "delay"]
        assert gw.obs.metric("gateway.ratelimit.delay.duration")

        # phase 2: over quota -> 429 with Retry-After + RateLimit-* headers, upstream untouched
        calls = up.call_count
        r = await _get(gw, h)
        assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
        assert int(r.headers["retry-after"]) >= 1
        assert r.headers["ratelimit-limit"] == "10" and r.headers["ratelimit-remaining"] == "0"
        assert "ratelimit-reset" in r.headers and "ratelimit-policy" in r.headers
        assert up.call_count == calls

        # phase 3: sustained abuse (ratio >= 1.5 for 5 s) -> block
        for _ in range(5):
            assert (await _get(gw, h)).status_code == 429
        gw.clock.advance(6)
        r = await _get(gw, h)
        assert r.status_code == 429 and r.json()["error"]["code"] == "client_blocked"
        gw.clock.advance(1)
        authz_before = len(gw.obs.audit("authz.decision"))
        blocked = await _get(gw, h)
        assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "client_blocked"
        assert int(blocked.headers["retry-after"]) >= 1
        assert len(gw.obs.audit("authz.decision")) == authz_before  # fast path: no policy work
        assert up.call_count == calls
        assert gw.obs.audit("ratelimit.client_blocked")

        # recovery after the cooldown (and with the old window fully rolled off)
        gw.clock.advance(130)
        h = gw.auth("billing:read")
        assert (await _get(gw, h)).status_code == 200


async def test_clients_are_isolated(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.get("http://billing.internal/invoices/42").respond(json={"ok": True})
        a = gw.auth("billing:read", client_id="client-a")
        for _ in range(16):  # ratio reaches block_at (1.5)
            await _get(gw, a)
        assert (await _get(gw, a)).status_code == 429
        gw.clock.advance(6)  # sustained long enough
        assert (await _get(gw, a)).json()["error"]["code"] == "client_blocked"
        assert (await _get(gw, a)).json()["error"]["code"] == "client_blocked"
        # client-b (unregistered -> default limits, own counters) is unaffected
        b = gw.auth("billing:read", client_id="client-b")
        for _ in range(15):
            r = await _get(gw, b)
            assert r.status_code == 200
        assert r.headers["ratelimit-limit"] == "1000"


# ------------------------------------------------------------------ observability


async def test_traces_metrics_audit_and_no_secrets(make_gateway: Any) -> None:
    async with make_gateway() as gw:
        gw.upstream.get("http://billing.internal/invoices/42").respond(json={"ok": True})
        gw.upstream.post(MCP).respond(json={"jsonrpc": "2.0", "id": 1, "result": {}})
        tok = gw.token("billing:read mcp:tools:read")
        h = {"Authorization": f"Bearer {tok}"}
        assert (
            await gw.client.get("/billing/invoices/42?api_key=SUPERSECRETQUERY", headers=h)
        ).status_code == 200
        r = await gw.client.post("/mcp/tools", headers=h, content=_rpc("tools/call", {"name": "get_x"}))
        assert r.status_code == 200
        await gw.client.get("/billing/invoices/42", headers={"Authorization": "Bearer not.a.jwt"})
        await gw.client.get("/billing/forbidden", headers=h)

        spans = gw.obs.spans.get_finished_spans()
        names = {s.name for s in spans}
        assert {
            "gateway.authn", "gateway.authz", "gateway.ratelimit", "gateway.proxy", "gateway.accounting"
        } <= names  # fmt: skip
        stage = next(s for s in spans if s.name == "gateway.authz")
        server = [s for s in spans if s.kind.name == "SERVER"]
        assert server and any(s.context.trace_id == stage.context.trace_id for s in server)
        assert any(s.kind.name == "CLIENT" for s in spans)  # httpx client span (traceparent injection)
        up = gw.upstream.calls[0].request
        assert "traceparent" in up.headers

        for metric in (
            "gateway.requests",
            "gateway.request.duration",
            "gateway.authz.decisions",
            "gateway.auth.failures",
            "gateway.ratelimit.decisions",
            "gateway.upstream.duration",
        ):
            assert gw.obs.metric(metric), metric
        statuses = {a["http.response.status_code"] for a, _ in gw.obs.metric("gateway.requests")}
        assert {200, 401, 403} <= {int(s) for s in statuses}

        types = {e["event"] for e in gw.obs.audit()}
        assert {"authz.decision", "authn.failure"} <= types
        allow = [e for e in gw.obs.audit("authz.decision") if e["decision"] == "allow"]
        assert allow and allow[0]["client_id"] == "client-a" and allow[0]["rule_id"] == "billing-read"
        assert any(r["event"] == "request.completed" for r in _app_logs(gw))

        blob = gw.obs.everything()
        for secret in (tok, "SUPERSECRETQUERY", MCP_SECRET, LLM_SECRET, "not.a.jwt"):
            assert secret not in blob
        sig = tok.split(".")[2]
        assert sig not in blob and tok.split(".")[1] not in blob


def _app_logs(gw: Any) -> list[dict[str, Any]]:
    out = []
    for line in gw.obs.stdout.getvalue().splitlines():
        if line.startswith("{"):
            rec = json.loads(line)
            if rec.get("stream") != "audit":
                out.append(rec)
    return out
