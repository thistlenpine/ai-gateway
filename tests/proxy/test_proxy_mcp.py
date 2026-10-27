from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from proxy_helpers import AsyncStream, collect, make_proxy, request, route, upstream_cfg

from ai_gateway.errors import BadRequestError
from ai_gateway.models import RouteMatch, UpstreamKind
from ai_gateway.proxy import ProxyResponse
from ai_gateway.proxy.mcp import McpAdapter

R = RouteMatch("u", UpstreamKind.MCP, "/mcp", "/")
A = McpAdapter()
CFG = upstream_cfg(UpstreamKind.MCP)
ALLOW = {"read_file", "list_dir"}


def allowed(name: str) -> bool:
    return name in ALLOW


def rpc(method: str, **params: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


def tools_result(*names: str) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": n, "description": "d"} for n in names]}}


# ------------------------------------------------------------------ classify


def test_classify_tools_call_and_methods() -> None:
    res = A.classify(R, "POST", json.dumps(rpc("tools/call", name="read_file", arguments={})).encode())
    assert [(r.mcp_method, r.tool) for r in res] == [("tools/call", "read_file")]
    res = A.classify(R, "POST", json.dumps(rpc("tools/list")).encode())
    assert res[0].mcp_method == "tools/list" and res[0].tool is None and res[0].upstream == "u"
    res = A.classify(R, "POST", b'{"jsonrpc":"2.0","method":"notifications/initialized"}')
    assert res[0].mcp_method == "notifications/initialized"


def test_classify_batch_every_message() -> None:
    body = json.dumps([rpc("ping"), rpc("tools/call", name="a"), rpc("tools/call", name="b")]).encode()
    res = A.classify(R, "POST", body)
    assert [(r.mcp_method, r.tool) for r in res] == [("ping", None), ("tools/call", "a"), ("tools/call", "b")]


@pytest.mark.parametrize("m", ["GET", "DELETE"])
def test_classify_get_delete(m: str) -> None:
    res = A.classify(R, m, None)
    assert len(res) == 1 and res[0].mcp_method is None and res[0].method == m


@pytest.mark.parametrize(
    "body",
    [
        None,
        b"",
        b"not json",
        b"[]",
        b"123",
        b'"x"',
        b"[1]",
        b"{}",
        b'{"method": 5}',
        b'{"method": ""}',
        b'{"jsonrpc":"2.0","id":1,"result":{}}',  # client response: unsupported -> deny
        b'{"method":"tools/call"}',
        b'{"method":"tools/call","params":{"name":5}}',
        b'{"method":"tools/call","params":[]}',
        b'{"method":"tools/list","method":"tools/call","params":{"name":"x"}}',  # duplicate keys
        b'{"method":"tools/call","params":{"name":"ok","name":"evil"}}',
        b"\xff\xfe",
        b'{"method":"ping","x":NaN}',
        b"[" * 5000 + b"]" * 5000,
        json.dumps([rpc("ping")] * 101).encode(),
    ],
)
def test_classify_rejects_malformed(body: bytes | None) -> None:
    with pytest.raises(BadRequestError):
        A.classify(R, "POST", body)


def test_classify_rejects_other_verbs() -> None:
    with pytest.raises(BadRequestError):
        A.classify(R, "PUT", b"{}")


# ------------------------------------------------------------------ tools/list filtering


async def run(
    resp_factory: Callable[[httpx.Request], httpx.Response],
    tool_allowed: Callable[[str], bool] | None = allowed,
) -> tuple[ProxyResponse, bytes]:
    proxy = make_proxy(resp_factory, CFG)
    resp = await proxy.forward(
        route(UpstreamKind.MCP, "/"),
        request("POST", "/", body=b"{}", headers=[("content-type", "application/json")]),
        tool_allowed=tool_allowed,
    )
    return resp, await collect(resp)


async def test_tools_list_json_filtered() -> None:
    doc = tools_result("read_file", "rm_rf", "list_dir", "exec")
    resp, body = await run(lambda r: httpx.Response(200, json=doc, headers={"mcp-session-id": "s1"}))
    names = [t["name"] for t in json.loads(body)["result"]["tools"]]
    assert names == ["read_file", "list_dir"]
    assert ("mcp-session-id", "s1") in [(k.lower(), v) for k, v in resp.headers]
    assert "content-length" not in {k.lower() for k, _ in resp.headers}


async def test_tools_list_batch_json_filtered_and_malformed_entries_dropped() -> None:
    r2 = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"tools": [{"name": "exec"}, {"nope": 1}, "str", {"name": "read_file"}]},
    }
    other = {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "tools"}]}}
    _, body = await run(lambda r: httpx.Response(200, json=[tools_result("exec"), r2, other]))
    out = json.loads(body)
    assert out[0]["result"]["tools"] == []
    assert [t["name"] for t in out[1]["result"]["tools"]] == ["read_file"]
    assert out[2] == other


async def test_unfiltered_json_is_byte_identical() -> None:
    raw = b'{"jsonrpc": "2.0",  "id": 1, "result": {"tools": [{"name": "read_file"}]}}'
    _, body = await run(
        lambda r: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    )
    assert body == raw


async def test_tools_call_result_and_errors_untouched() -> None:
    doc = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": "x"}], "isError": False},
    }
    _, body = await run(lambda r: httpx.Response(200, json=doc))
    assert json.loads(body) == doc
    _, body = await run(lambda r: httpx.Response(500, content=b"<html>tools</html>"))
    assert body == b"<html>tools</html>"


async def test_empty_202_and_no_callback_passthrough() -> None:
    resp, body = await run(lambda r: httpx.Response(202, headers={"content-type": "application/json"}))
    assert resp.status == 202 and body == b""
    doc = tools_result("exec")
    _, body = await run(lambda r: httpx.Response(200, json=doc), tool_allowed=None)
    assert json.loads(body) == doc


async def test_invalid_json_from_upstream_fails_closed() -> None:
    from ai_gateway.errors import UpstreamError

    with pytest.raises(UpstreamError):
        await run(
            lambda r: httpx.Response(200, content=b"{broken", headers={"content-type": "application/json"})
        )


def sse(doc: object, *, event: str | None = "message", eid: str | None = "7") -> bytes:
    s = ""
    if eid:
        s += f"id: {eid}\n"
    if event:
        s += f"event: {event}\n"
    return (s + f"data: {json.dumps(doc)}\n\n").encode()


async def test_tools_list_sse_filtered_other_events_byte_identical() -> None:
    progress = b'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n'
    keepalive = b": ping\n\n"
    ev = sse(tools_result("read_file", "exec"))
    stream = progress + keepalive + ev
    # split mid-event and mid-line to exercise reassembly
    parts = [stream[:30], stream[30 : len(progress) + 5], stream[len(progress) + 5 :]]
    resp, body = await run(
        lambda r: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=AsyncStream(parts)
        )
    )
    assert resp.is_stream
    assert body.startswith(progress + keepalive)
    tail = body[len(progress + keepalive) :].decode()
    assert tail.startswith("id: 7\nevent: message\ndata: ") and tail.endswith("\n\n")
    data = json.loads(tail.split("data: ", 1)[1])
    assert [t["name"] for t in data["result"]["tools"]] == ["read_file"]


async def test_sse_events_emitted_as_they_arrive() -> None:
    import asyncio

    gate = asyncio.Event()

    class Gated(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield sse(tools_result("read_file"))
            await gate.wait()
            yield sse(tools_result("exec"))

    proxy = make_proxy(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Gated()), CFG
    )
    resp = await proxy.forward(route(UpstreamKind.MCP, "/"), request("GET", "/"), tool_allowed=allowed)
    it = resp.body.__aiter__()
    first = await asyncio.wait_for(it.__anext__(), 2)
    assert b"read_file" in first
    gate.set()
    second = await asyncio.wait_for(it.__anext__(), 2)
    assert b'"tools":[]' in second
    await resp.aclose()


async def test_sse_crlf_framing() -> None:
    payload = json.dumps(tools_result("exec", "read_file"))
    raw = f"event: message\r\ndata: {payload}\r\n\r\n".encode()
    _, body = await run(
        lambda r: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=AsyncStream([raw[:-1], raw[-1:]])
        )
    )
    assert b"exec" not in body and b"read_file" in body


async def test_mcp_session_headers_forwarded() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    proxy = make_proxy(handler, CFG)
    await collect(
        await proxy.forward(
            route(UpstreamKind.MCP, "/"),
            request(
                "POST",
                "/",
                body=b"{}",
                headers=[
                    ("Mcp-Session-Id", "abc"),
                    ("Last-Event-ID", "9"),
                    ("Accept", "application/json, text/event-stream"),
                ],
            ),
            tool_allowed=allowed,
        )
    )
    assert seen[0].headers["mcp-session-id"] == "abc" and seen[0].headers["last-event-id"] == "9"
