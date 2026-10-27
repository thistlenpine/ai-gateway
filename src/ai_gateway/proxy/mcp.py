"""MCP streamable-HTTP adapter (kind=mcp). OWNER: proxy.

Transport facts: POST carries one JSON-RPC message or a batch and is answered with
application/json OR text/event-stream; GET opens a server->client SSE stream; DELETE ends the
session; `Mcp-Session-Id` and `Last-Event-ID` headers must pass through untouched.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

from ai_gateway.config import UpstreamConfig
from ai_gateway.errors import BadRequestError, UpstreamError
from ai_gateway.models import Resource, RouteMatch, UpstreamKind
from ai_gateway.proxy._body import read_bounded, with_body
from ai_gateway.proxy._json import loads_strict, parse_body
from ai_gateway.proxy.base import ProxyRequest, ProxyResponse, UsageCallback
from ai_gateway.proxy.headers import get_header
from ai_gateway.proxy.streaming import SseEvent, SseParser, aclose_quietly

_MAX_BATCH = 100
_BUFFER_LIMIT = 8 * 1024 * 1024


def _resource(route: RouteMatch, msg: Any) -> Resource:
    if not isinstance(msg, dict):
        raise BadRequestError("Invalid JSON-RPC message")
    method = msg.get("method")
    if not isinstance(method, str) or not method:
        # Client->server *responses* (sampling/elicitation replies) are not supported: deny.
        raise BadRequestError("Invalid JSON-RPC message")
    tool: str | None = None
    if method == "tools/call":
        params = msg.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        if not isinstance(name, str) or not name:
            raise BadRequestError("tools/call requires a tool name")
        tool = name
    return Resource(UpstreamKind.MCP, route.upstream, "POST", route.relative_path, method, tool)


def filter_tools_message(msg: Any, allowed: Callable[[str], bool]) -> bool:
    """In-place: drop disallowed tools from a `result.tools` list. Returns True if changed.
    Entries without a string name are dropped too (fail closed)."""
    if not isinstance(msg, dict):
        return False
    result = msg.get("result")
    if not isinstance(result, dict):
        return False
    tools = result.get("tools")
    if not isinstance(tools, list):
        return False
    kept = [t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str) and allowed(t["name"])]
    if len(kept) == len(tools):
        return False
    result["tools"] = kept
    return True


def _filter_json(body: bytes, allowed: Callable[[str], bool]) -> bytes:
    try:
        doc = loads_strict(body)
    except ValueError:
        raise UpstreamError("Invalid upstream response") from None
    msgs = doc if isinstance(doc, list) else [doc]
    changed = False
    for m in msgs:
        changed = filter_tools_message(m, allowed) or changed
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode() if changed else body


def _rewrite_event(ev: SseEvent, allowed: Callable[[str], bool]) -> bytes:
    if b'"tools"' not in ev.raw or ev.event not in (None, "message") or not ev.data:
        return ev.raw
    try:
        doc = loads_strict(ev.data)
    except ValueError:
        return ev.raw
    msgs = doc if isinstance(doc, list) else [doc]
    changed = False
    for m in msgs:
        changed = filter_tools_message(m, allowed) or changed
    if not changed:
        return ev.raw
    new_data = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    out: list[str] = []
    wrote = False
    for line in re.split(r"\r\n|\r|\n", ev.raw.decode("utf-8", errors="replace")):
        if line.startswith("data"):
            if not wrote:
                out.append(f"data: {new_data}")
                wrote = True
            continue
        if line:
            out.append(line)
    return ("\n".join(out) + "\n\n").encode()


async def _filter_json_body(
    source: AsyncIterator[bytes], allowed: Callable[[str], bool]
) -> AsyncIterator[bytes]:
    try:
        raw = await read_bounded(source, _BUFFER_LIMIT)
        yield _filter_json(raw, allowed) if raw.strip() else raw
    finally:
        await aclose_quietly(source)


async def _filter_sse_body(
    source: AsyncIterator[bytes], allowed: Callable[[str], bool]
) -> AsyncIterator[bytes]:
    parser = SseParser()
    try:
        async for chunk in source:
            for ev in parser.feed(chunk):
                yield _rewrite_event(ev, allowed)
        for ev in parser.finish():
            yield _rewrite_event(ev, allowed)
    finally:
        await aclose_quietly(source)


class McpAdapter:
    kind = UpstreamKind.MCP

    def classify(self, route: RouteMatch, method: str, body: bytes | None) -> list[Resource]:
        """POST: parse JSON-RPC (object or array); each message -> Resource(mcp_method=method,
        tool=params.name for tools/call). Non-JSON-RPC or empty batch -> BadRequestError.
        GET/DELETE: single Resource with mcp_method=None."""
        verb = method.upper()
        if verb in ("GET", "DELETE"):
            return [Resource(UpstreamKind.MCP, route.upstream, verb, route.relative_path)]
        if verb != "POST":
            raise BadRequestError("Method not supported")
        doc = parse_body(body)
        if isinstance(doc, list):
            if not doc or len(doc) > _MAX_BATCH:
                raise BadRequestError("Invalid JSON-RPC batch")
            return [_resource(route, m) for m in doc]
        return [_resource(route, doc)]

    def estimate_tokens(self, body: bytes | None, cap: int | None = None) -> int:
        return 0

    def prepare(self, upstream: UpstreamConfig, req: ProxyRequest) -> ProxyRequest:
        return req

    def process_response(
        self,
        resp: ProxyResponse,
        *,
        tool_allowed: Callable[[str], bool] | None,
        on_usage: UsageCallback | None,
    ) -> ProxyResponse:
        """Filter tools/list results through tool_allowed (JSON body or SSE `message` events);
        everything else is passed through byte-for-byte."""
        if tool_allowed is None or not 200 <= resp.status < 300:
            return resp
        if resp.is_stream:
            return with_body(resp, _filter_sse_body(resp.body, tool_allowed))
        ctype = (get_header(resp.headers, "content-type") or "").lower()
        if "json" in ctype:
            return with_body(resp, _filter_json_body(resp.body, tool_allowed))
        return resp
