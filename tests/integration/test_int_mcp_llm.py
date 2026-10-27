"""MCP session flow and LLM streaming with usage accounting, over real sockets."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from integration.servers import LLM_SECRET, MCP_SECRET, OAI_SECRET, SESSION_ID, UpstreamState

MCP = "/mcp/tools"
LLM = "/llm/anthropic/v1/messages"
OAI = "/llm/openai/v1/chat/completions"


def rpc(method: str, params: dict[str, Any] | None = None, rid: int | None = 1) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if rid is not None:
        msg["id"] = rid
    return msg


def sse_data(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


# --------------------------------------------------------------------------- MCP


@pytest.mark.parametrize("via_sse", [False, True], ids=["json", "sse"])
async def test_mcp_full_session_flow(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState, via_sse: bool
) -> None:
    upstream_state.tools_list_via_sse = via_sse
    gw = make_gw()
    h = gw.headers("mcp:ro")

    # 1. initialize: the upstream mints the session id; it must reach the client untouched
    r = await http.post(f"{gw.url}{MCP}", json=rpc("initialize"), headers=h)
    assert r.status_code == 200 and r.headers["mcp-session-id"] == SESSION_ID
    sh = {**h, "Mcp-Session-Id": SESSION_ID}

    # 2. notification -> 202 passthrough
    r = await http.post(f"{gw.url}{MCP}", json=rpc("notifications/initialized", rid=None), headers=sh)
    assert r.status_code == 202

    # 3. tools/list is filtered to what the principal may call (both JSON and SSE bodies)
    r = await http.post(f"{gw.url}{MCP}", json=rpc("tools/list", rid=2), headers=sh)
    assert r.status_code == 200
    doc = sse_data(r.text)[0] if via_sse else r.json()
    assert [t["name"] for t in doc["result"]["tools"]] == ["get_weather", "search_docs"]

    # 4. allowed call goes through, denied one never reaches the upstream
    r = await http.post(f"{gw.url}{MCP}", json=rpc("tools/call", {"name": "get_weather"}, 3), headers=sh)
    assert r.status_code == 200 and "ran get_weather" in r.text
    calls = upstream_state.count("/mcp")
    r = await http.post(
        f"{gw.url}{MCP}", json=rpc("tools/call", {"name": "delete_everything"}, 4), headers=sh
    )
    assert r.status_code == 403 and r.json()["error"]["code"] == "access_denied"
    assert upstream_state.count("/mcp") == calls

    # 5. server->client stream with resumption header
    async with http.stream("GET", f"{gw.url}{MCP}", headers={**sh, "Last-Event-ID": "7"}) as s:
        assert s.status_code == 200 and s.headers["content-type"].startswith("text/event-stream")
        text = "".join([c async for c in s.aiter_text()])
    msgs = sse_data(text)
    assert [m["params"]["n"] for m in msgs] == [8, 9, 10] and msgs[0]["params"]["last"] == "7"

    # 6. session end
    assert (await http.delete(f"{gw.url}{MCP}", headers=sh)).status_code == 200

    # what the upstream saw: gateway credential, session header, never the caller's token
    for seen in (s for s in upstream_state.requests if s.path.startswith("/mcp")):
        assert seen.headers["authorization"] == f"Bearer {MCP_SECRET}"
        assert h["Authorization"].split()[1] not in json.dumps(seen.headers)
    assert upstream_state.last("/mcp").headers["mcp-session-id"] == SESSION_ID


async def test_mcp_batch_requires_every_message_allowed(make_gw: Any, http: httpx.AsyncClient) -> None:
    gw = make_gw()
    h = {**gw.headers("mcp:ro"), "Mcp-Session-Id": SESSION_ID}
    ok = await http.post(
        f"{gw.url}{MCP}", json=[rpc("ping", rid=1), rpc("tools/call", {"name": "get_x"}, 2)], headers=h
    )
    assert ok.status_code == 200 and len(ok.json()) == 2
    bad = await http.post(
        f"{gw.url}{MCP}", json=[rpc("ping", rid=1), rpc("tools/call", {"name": "admin_reset"}, 2)], headers=h
    )
    assert bad.status_code == 403
    assert (await http.post(f"{gw.url}{MCP}", content=b"{not json", headers=h)).status_code == 400


# --------------------------------------------------------------------------- LLM


def llm_req(*, stream: bool, max_tokens: int = 200, model: str = "claude-sonnet-4") -> dict[str, Any]:
    return {"model": model, "max_tokens": max_tokens, "stream": stream,
            "messages": [{"role": "user", "content": "x" * 40}]}  # fmt: skip


async def test_llm_streaming_passthrough_and_usage_accounting(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState
) -> None:
    gw = make_gw(
        clients={"int-client": {"limits": {"tokens_per_window": 1000, "token_window_seconds": 3600}}}
    )
    h = gw.headers("llm")
    upstream_state.llm_tokens = (25, 40)
    async with http.stream("POST", f"{gw.url}{LLM}", json=llm_req(stream=True), headers=h) as r:
        assert r.status_code == 200
        # estimate = ceil(40/4) + 200 = 210 is reserved while streaming
        reserved = sum(gw.store_keys("rl:int-client:tok:").values())
        body = "".join([c async for c in r.aiter_text()])
    events = sse_data(body)
    assert events[0]["type"] == "message_start" and events[-1]["type"] == "message_stop"
    assert sum(e["type"] == "content_block_delta" for e in events) == 4
    assert reserved == 210
    seen = upstream_state.last("/anthropic")
    assert seen.headers["x-api-key"] == LLM_SECRET and "authorization" not in seen.headers
    # after completion the counter is reconciled to actual usage: 25 + 40
    import asyncio

    for _ in range(100):
        if sum(gw.store_keys("rl:int-client:tok:").values()) == 65:
            break
        await asyncio.sleep(0.02)
    assert sum(gw.store_keys("rl:int-client:tok:").values()) == 65


async def test_llm_non_streaming_usage_and_max_tokens_clamp(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState
) -> None:
    gw = make_gw(
        clients={"int-client": {"limits": {"tokens_per_window": 1000, "token_window_seconds": 3600}}}
    )
    r = await http.post(
        f"{gw.url}{LLM}", json=llm_req(stream=False, max_tokens=900), headers=gw.headers("llm")
    )
    assert r.status_code == 200 and r.json()["usage"] == {"input_tokens": 25, "output_tokens": 40}
    assert json.loads(upstream_state.last("/anthropic").body)["max_tokens"] == 500  # grant cap enforced
    assert sum(gw.store_keys("rl:int-client:tok:").values()) == 65
    denied = await http.post(
        f"{gw.url}{LLM}", json=llm_req(stream=False, model="gpt-4o"), headers=gw.headers("llm")
    )
    assert denied.status_code == 403
    nomodel = await http.post(f"{gw.url}{LLM}", json={"messages": []}, headers=gw.headers("llm"))
    assert nomodel.status_code == 400


async def test_token_quota_exhaustion_and_recovery_after_refund(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState
) -> None:
    import asyncio

    gw = make_gw(clients={"int-client": {"limits": {"tokens_per_window": 750, "token_window_seconds": 3600}}})
    h = gw.headers("llm")
    codes = []
    for i in range(8):  # each call really uses 65 tokens but reserves 210 up front
        codes.append((await http.post(f"{gw.url}{LLM}", json=llm_req(stream=False), headers=h)).status_code)
        for _ in range(100):  # the settle step runs in the response finalizer, just after the body
            if sum(gw.store_keys("rl:int-client:tok:").values()) == 65 * (i + 1):
                break
            await asyncio.sleep(0.02)
    assert codes == [200] * 8  # 8*65 = 520 settled
    r = await http.post(f"{gw.url}{LLM}", json=llm_req(stream=False, max_tokens=500), headers=h)
    assert r.status_code == 429 and r.json()["error"]["code"] == "token_quota_exceeded"  # 520 + 510 > 750
    assert int(r.headers["retry-after"]) >= 1
    # a rejected reservation is rolled back (never inflates the counter) and smaller calls still fit
    assert sum(gw.store_keys("rl:int-client:tok:").values()) == 520
    assert (await http.post(f"{gw.url}{LLM}", json=llm_req(stream=False), headers=h)).status_code == 200


async def test_openai_stream_gets_include_usage_injected_and_is_accounted(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState
) -> None:
    import asyncio

    gw = make_gw(
        clients={"int-client": {"limits": {"tokens_per_window": 5000, "token_window_seconds": 3600}}}
    )
    body = {
        "model": "gpt-4o",
        "stream": True,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    r = await http.post(f"{gw.url}{OAI}", json=body, headers=gw.headers("llm"))
    assert r.status_code == 200 and "[DONE]" in r.text
    seen = upstream_state.last("/openai")
    assert json.loads(seen.body)["stream_options"] == {"include_usage": True}
    assert seen.headers["authorization"] == f"Bearer {OAI_SECRET}"
    for _ in range(100):
        if sum(gw.store_keys("rl:int-client:tok:").values()) == 65:
            break
        await asyncio.sleep(0.02)
    assert sum(gw.store_keys("rl:int-client:tok:").values()) == 65  # usage from the injected final chunk


async def test_client_abort_of_llm_stream_closes_upstream_and_settles_tokens(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState
) -> None:
    import asyncio

    upstream_state.llm_tokens = (25, 4000)  # long stream: 400 content deltas
    gw = make_gw(
        clients={"int-client": {"limits": {"tokens_per_window": 100_000, "token_window_seconds": 3600}}}
    )
    async with http.stream(
        "POST", f"{gw.url}{LLM}", json=llm_req(stream=True), headers=gw.headers("llm")
    ) as r:
        async for chunk in r.aiter_bytes():
            if b"message_start" in chunk:
                break  # usage (input 25) has been observed: abort now
    for _ in range(200):
        if upstream_state.sse_open == 0:
            break
        await asyncio.sleep(0.02)
    assert upstream_state.sse_open == 0 and upstream_state.sse_aborted == 1
    for _ in range(100):  # accounting runs in the response finalizer
        if (
            sum(gw.store_keys("rl:int-client:tok:").values()) != 210
        ):  # 10 + max_tokens 200 reserved up front: wait until it is settled
            break
        await asyncio.sleep(0.02)
    used = sum(gw.store_keys("rl:int-client:tok:").values())
    assert used == 26, used  # settled at the observed usage (25 in + 1 out), not the 4200 estimate


async def test_token_estimate_uses_the_grant_clamped_max_tokens(
    make_gw: Any, http: httpx.AsyncClient, upstream_state: UpstreamState
) -> None:
    """The pre-flight estimate uses max_tokens after the grant clamp (cap 500), because that is all
    the upstream will ever see: an oversized request must not be refused by a small quota."""
    gw = make_gw(
        clients={"int-client": {"limits": {"tokens_per_window": 1000, "token_window_seconds": 3600}}}
    )
    r = await http.post(
        f"{gw.url}{LLM}", json=llm_req(stream=False, max_tokens=99_999), headers=gw.headers("llm")
    )
    assert r.status_code != 429
    assert upstream_state.count("/anthropic") == 1
